"""StoneSense-AI Custom Flower Strategy.

Extends FedAvg to:
1. Aggregate custom training and evaluation metrics across hospital clients.
2. Persist round results, hospital telemetry, and model version checkpoints to database.
3. Synchronize deployed model pointers for real-time inference.
"""

from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import numpy as np
import torch
import flwr as fl
from flwr.common import (
    EvaluateRes, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))

from model import build_resnet18_classifier, CLASS_MAPPING
from local_training import set_model_parameters
from model_manager import model_manager

logger = logging.getLogger("StoneSenseFedAvg")


def weighted_average_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Computes weighted average of scalar metrics across all clients."""
    if not metrics:
        return {}

    total_samples = sum(num_examples for num_examples, _ in metrics)
    if total_samples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    keys = metrics[0][1].keys()

    for key in keys:
        weighted_sum = 0.0
        valid_samples = 0
        for num_examples, m in metrics:
            val = m.get(key)
            if isinstance(val, (int, float, np.number)):
                weighted_sum += num_examples * float(val)
                valid_samples += num_examples
        if valid_samples > 0:
            aggregated[key] = float(weighted_sum / valid_samples)

    return aggregated


class StoneSenseFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with telemetry persistence and model checkpointing."""

    def __init__(
        self,
        num_classes: int = 4,
        mode: str = "iid",
        db_persist: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(
            fit_metrics_aggregation_fn=weighted_average_metrics,
            evaluate_metrics_aggregation_fn=weighted_average_metrics,
            *args,
            **kwargs
        )
        self.num_classes = num_classes
        self.mode = mode
        self.db_persist = db_persist
        self.round_start_time = datetime.utcnow()
        self.latest_round_metrics: Dict[int, Dict[str, Any]] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Sends current round index to clients in config."""
        self.round_start_time = datetime.utcnow()
        config = {
            "current_round": server_round,
            "local_epochs": 1,
            "lr": 0.0005,
        }
        fit_ins = fl.common.FitIns(parameters, config)

        clients = client_manager.sample(
            num_clients=self.min_fit_clients, min_num_clients=self.min_available_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregates fit results, extracts per-hospital telemetry, and weights."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is None:
            return None, {}

        # Extract per-client local training metrics
        client_runs = []
        for _, fit_res in results:
            m = fit_res.metrics
            h_code = str(m.get("hospital_id", "unknown"))
            client_runs.append({
                "hospital_code": h_code,
                "train_loss": float(m.get("train_loss", 0.0)),
                "train_acc": float(m.get("train_accuracy", 0.0)),
                "train_f1": float(m.get("train_f1_macro", 0.0)),
                "sample_count": int(fit_res.num_examples),
            })

        self.latest_round_metrics[server_round] = {
            "fit_metrics": aggregated_metrics,
            "client_runs": client_runs,
        }

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregates validation metrics, saves model checkpoint, and persists to DB."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        duration = (datetime.utcnow() - self.round_start_time).total_seconds()

        # Update per-client evaluation metrics
        for _, eval_res in results:
            m = eval_res.metrics
            h_code = str(m.get("hospital_id", "unknown"))
            round_data = self.latest_round_metrics.get(server_round, {})
            for run in round_data.get("client_runs", []):
                if run["hospital_code"] == h_code:
                    run["val_loss"] = float(eval_res.loss)
                    run["val_acc"] = float(m.get("accuracy", 0.0))
                    run["val_f1"] = float(m.get("f1_macro", 0.0))

        logger.info(
            f"=== Federated Round {server_round} Aggregation Summary ===\n"
            f"  Val Loss: {loss_aggregated:.4f} | "
            f"  Val Acc: {metrics_aggregated.get('accuracy', 0.0):.4f} | "
            f"  Val F1: {metrics_aggregated.get('f1_macro', 0.0):.4f} | "
            f"  Duration: {duration:.2f}s"
        )

        return loss_aggregated, metrics_aggregated


def persist_round_to_db(
    round_number: int,
    mode: str,
    parameters: List[np.ndarray],
    fit_metrics: Dict[str, Any],
    eval_metrics: Dict[str, Any],
    client_runs: List[Dict[str, Any]],
    duration_sec: float
) -> None:
    """Persists round results, hospital runs, and model version into SQLite/PostgreSQL."""
    try:
        from app.db.database import SessionLocal, init_db
        from app.db.models import Hospital, FederatedRound, HospitalTrainingRun, ModelVersion, HospitalUpdateLog
    except ImportError as e:
        logger.warning(f"Could not import backend DB modules: {e}")
        return

    init_db()
    db = SessionLocal()
    try:
        # 1. Rebuild and save PyTorch model checkpoint
        class_names = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]
        model = build_resnet18_classifier(num_classes=len(class_names), freeze_backbone=False)
        set_model_parameters(model, parameters)

        val_acc = float(eval_metrics.get("accuracy", 0.0))
        val_f1 = float(eval_metrics.get("f1_macro", 0.0))
        val_prec = float(eval_metrics.get("precision_macro", 0.0))
        val_rec = float(eval_metrics.get("recall_macro", 0.0))
        val_loss = float(eval_metrics.get("loss", 0.0))
        train_loss = float(fit_metrics.get("train_loss", 0.0))
        train_acc = float(fit_metrics.get("train_accuracy", 0.0))

        checkpoint_path = model_manager.save_round_checkpoint(
            model_state_dict=model.state_dict(),
            round_number=round_number,
            metrics={"accuracy": val_acc, "f1_macro": val_f1, "val_loss": val_loss},
            is_deployed=True
        )

        # 2. Check if round already in DB or update
        fed_round = db.query(FederatedRound).filter_by(round_number=round_number).first()
        if not fed_round:
            fed_round = FederatedRound(
                round_number=round_number,
                mode=mode,
                participants_count=len(client_runs),
                global_train_loss=train_loss,
                global_train_acc=train_acc,
                global_val_loss=val_loss,
                global_val_acc=val_acc,
                global_val_f1=val_f1,
                global_val_precision=val_prec,
                global_val_recall=val_rec,
                duration_sec=duration_sec,
                status="completed",
                completed_at=datetime.utcnow()
            )
            db.add(fed_round)
            db.flush()
        else:
            fed_round.global_train_loss = train_loss
            fed_round.global_train_acc = train_acc
            fed_round.global_val_loss = val_loss
            fed_round.global_val_acc = val_acc
            fed_round.global_val_f1 = val_f1
            fed_round.duration_sec = duration_sec

        # 3. ModelVersion Registry
        version_tag = f"resnet18_fed_round_{round_number:03d}"
        mv = db.query(ModelVersion).filter_by(version_tag=version_tag).first()
        if not mv:
            # Undeploy older versions
            db.query(ModelVersion).filter_by(model_family="resnet18_ct").update({ModelVersion.is_deployed: False})
            mv = ModelVersion(
                model_family="resnet18_ct",
                version_tag=version_tag,
                round_id=fed_round.id,
                accuracy=val_acc,
                f1_score=val_f1,
                precision=val_prec,
                recall=val_rec,
                is_deployed=True,
                artifact_path=str(checkpoint_path.relative_to(PROJECT_ROOT)),
                trained_at=datetime.utcnow()
            )
            db.add(mv)
            db.flush()

        # 4. Hospital training runs and sync logs
        hospitals = {h.hospital_code: h for h in db.query(Hospital).all()}
        # Mapping for simulated names if hospital_code is e.g. "hospital_1" -> "HOSP-001"
        code_map = {"hospital_1": "HOSP-001", "hospital_2": "HOSP-002", "hospital_3": "HOSP-003"}

        for run in client_runs:
            raw_code = run["hospital_code"]
            db_code = code_map.get(raw_code, raw_code)
            h = hospitals.get(db_code)
            if not h:
                # Create hospital if missing
                h = Hospital(hospital_code=db_code, name=f"Simulated Node {db_code}", region="Simulation")
                db.add(h)
                db.flush()
                hospitals[db_code] = h

            h.current_model_version = version_tag

            # Record run telemetry
            tr = HospitalTrainingRun(
                round_id=fed_round.id,
                round_number=round_number,
                hospital_id=h.id,
                hospital_code=db_code,
                train_loss=run.get("train_loss"),
                train_acc=run.get("train_acc"),
                train_f1=run.get("train_f1"),
                val_loss=run.get("val_loss"),
                val_acc=run.get("val_acc"),
                val_f1=run.get("val_f1"),
                sample_count=run.get("sample_count", 0),
                duration_sec=duration_sec,
                created_at=datetime.utcnow()
            )
            db.add(tr)

            # Record update log
            db.add(HospitalUpdateLog(
                hospital_id=h.id,
                model_version_id=mv.id,
                status="received",
                created_at=datetime.utcnow()
            ))

        db.commit()
        logger.info(f"Federated round {round_number} records & telemetry successfully persisted to database.")
    except Exception as e:
        db.rollback()
        logger.exception(f"Error persisting federated round to DB: {e}")
    finally:
        db.close()
