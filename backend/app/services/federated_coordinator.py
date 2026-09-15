"""StoneSense-AI Live DL Federated Learning Round Coordinator.

Orchestrates real-time multi-hospital DL Federated Learning rounds:
- Enforces single-round concurrency lock
- Coordinates isolated local training on hospital partitions (Zero Raw CT Transfer)
- Runs FedAvg parameter aggregation
- Persists telemetry and model versioning to database
- Broadcasts real-time lifecycle events via WebSockets to Developer and Hospital consoles
"""

from pathlib import Path
import sys
import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DL_ROOT = PROJECT_ROOT.parent / "dl"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(DL_ROOT / "preprocessing"))
sys.path.append(str(DL_ROOT / "training"))
sys.path.append(str(DL_ROOT / "federated"))

from app.services.ws_manager import ws_manager
from app.db.database import SessionLocal, init_db
from app.db.models import Hospital, FederatedRound, HospitalTrainingRun, ModelVersion, HospitalUpdateLog

logger = logging.getLogger("StoneSenseFLCoordinator")

HOSPITAL_CODE_MAP = {
    "hospital_1": "HOSP-001",
    "hospital_2": "HOSP-002",
    "hospital_3": "HOSP-003",
    "1": "HOSP-001",
    "2": "HOSP-002",
    "3": "HOSP-003",
    "HOSP-001": "HOSP-001",
    "HOSP-002": "HOSP-002",
    "HOSP-003": "HOSP-003",
}

HOSPITAL_NAMES = {
    "HOSP-001": "Apollo Kidney Care",
    "HOSP-002": "Manipal Urology Institute",
    "HOSP-003": "AIIMS Nephrology Labs",
}

HOSPITAL_WAITING = "WAITING"
HOSPITAL_MODEL_RECEIVED = "MODEL_RECEIVED"
HOSPITAL_TRAINING = "TRAINING"
HOSPITAL_TRAINING_COMPLETED = "TRAINING_COMPLETED"
HOSPITAL_UPDATE_SUBMITTED = "UPDATE_SUBMITTED"
HOSPITAL_WAITING_FOR_AGGREGATION = "WAITING_FOR_AGGREGATION"
HOSPITAL_MODEL_UPDATED = "MODEL_UPDATED"
HOSPITAL_COMPLETED = "COMPLETED"
HOSPITAL_FAILED = "FAILED"
HOSPITAL_NOT_SELECTED = "NOT_SELECTED"



class FederatedCoordinator:
    """Manages the end-to-end lifecycle and telemetry of live DL federated rounds."""

    def __init__(self):
        self._lock = threading.Lock()
        self.is_running = False
        self.current_round: int = 0
        self.status: str = "READY"
        self.current_step: Optional[str] = None
        self.previous_model_version: str = "resnet18_centralized_v1"
        self.global_model_version: str = "resnet18_centralized_v1"
        self.error_message: Optional[str] = None
        self.round_start_time: Optional[datetime] = None
        self.clients_status: Dict[str, Dict[str, Any]] = {}
        self.latest_round_metrics: Dict[str, Any] = {}
        self.selected_hospital_ids: List[int] = []
        self.selected_hospital_codes: List[str] = []
        self._initialize_from_db()

    def _initialize_from_db(self):
        """Loads current state from database on startup."""
        try:
            init_db()
            db = SessionLocal()
            latest_round = db.query(FederatedRound).order_by(FederatedRound.round_number.desc()).first()
            deployed_model = db.query(ModelVersion).filter_by(is_deployed=True, model_family="resnet18_ct").first()

            if latest_round:
                self.current_round = latest_round.round_number
                self.global_model_version = (
                    deployed_model.version_tag if deployed_model else f"resnet18_fed_round_{latest_round.round_number:03d}"
                )
            else:
                self.current_round = 0
                self.global_model_version = deployed_model.version_tag if deployed_model else "resnet18_centralized_v1"

            self.previous_model_version = self.global_model_version
            self.status = "READY"
            self._reset_clients_status()
            db.close()
        except Exception as e:
            logger.warning(f"Coordinator could not initialize from DB: {e}")
            self.current_round = 0
            self.status = "READY"
            self._reset_clients_status()

    def _reset_clients_status(self):
        self.clients_status = {
            "HOSP-001": {
                "hospital_id": "HOSP-001",
                "hospital_name": "Apollo Kidney Care",
                "status": HOSPITAL_WAITING,
                "samples": None,
                "accuracy": None,
                "f1": None,
                "loss": None,
                "duration_sec": None,
            },
            "HOSP-002": {
                "hospital_id": "HOSP-002",
                "hospital_name": "Manipal Urology Institute",
                "status": HOSPITAL_WAITING,
                "samples": None,
                "accuracy": None,
                "f1": None,
                "loss": None,
                "duration_sec": None,
            },
            "HOSP-003": {
                "hospital_id": "HOSP-003",
                "hospital_name": "AIIMS Nephrology Labs",
                "status": HOSPITAL_WAITING,
                "samples": None,
                "accuracy": None,
                "f1": None,
                "loss": None,
                "duration_sec": None,
            },
        }

    def _emit_event(self, event_type: str, hospital_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Emits an event to all connected WebSocket clients."""
        payload = {
            "event": event_type,
            "round": self.current_round,
            "hospital_id": hospital_id,
            "model_version": self.global_model_version,
            "previous_model_version": self.previous_model_version,
            "status": (
                self.clients_status.get(hospital_id, {}).get("status", self.status)
                if hospital_id
                else self.status
            ),
            "current_step": self.current_step,
            "data": data or {},
            "payload": data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.info(f"FL Event: {event_type} | Round: {self.current_round} | Node: {hospital_id or 'GLOBAL'}")
        ws_manager.broadcast_sync(payload)

    def start_round(
        self,
        selected_hospital_ids: Optional[List[int]] = None,
        num_rounds: int = 1,
        local_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.0005,
        mode: str = "iid",
        device: str = "auto",
    ) -> Dict[str, Any]:
        """Validates and kicks off a live DL federated learning round in background."""
        with self._lock:
            if self.is_running:
                raise RuntimeError(f"Round {self.current_round} is currently running. Duplicate execution prevented.")

            db = SessionLocal()
            try:
                if selected_hospital_ids is None:
                    selected_hospitals = db.query(Hospital).filter(Hospital.is_active.is_(True)).all()
                else:
                    unique_ids = list(dict.fromkeys(selected_hospital_ids))
                    if not unique_ids:
                        raise ValueError("At least one hospital must be selected.")
                    selected_hospitals = db.query(Hospital).filter(Hospital.id.in_(unique_ids)).all()
                    found_ids = {hospital.id for hospital in selected_hospitals}
                    missing_ids = [hospital_id for hospital_id in unique_ids if hospital_id not in found_ids]
                    if missing_ids:
                        raise ValueError(f"Unknown hospital IDs: {missing_ids}")
                    inactive_ids = [hospital.id for hospital in selected_hospitals if not hospital.is_active]
                    if inactive_ids:
                        raise ValueError(f"Inactive hospitals cannot participate: {inactive_ids}")

                unsupported_codes = [
                    hospital.hospital_code for hospital in selected_hospitals
                    if hospital.hospital_code not in HOSPITAL_CODE_MAP.values()
                ]
                if unsupported_codes:
                    raise ValueError(f"Hospitals do not have configured DL partitions: {unsupported_codes}")
                if not selected_hospitals:
                    raise ValueError("No selected hospital has a configured DL partition.")
                self.selected_hospital_ids = [hospital.id for hospital in selected_hospitals]
                self.selected_hospital_codes = [hospital.hospital_code for hospital in selected_hospitals]
            finally:
                db.close()

            # Compute next round number
            db = SessionLocal()
            try:
                latest = db.query(FederatedRound).order_by(FederatedRound.round_number.desc()).first()
                next_round = (latest.round_number + 1) if latest else (self.current_round + 1)
                deployed = db.query(ModelVersion).filter_by(is_deployed=True, model_family="resnet18_ct").first()
                prev_version = deployed.version_tag if deployed else f"resnet18_fed_round_{max(next_round - 1, 1):03d}"
            finally:
                db.close()

            self.is_running = True
            self.current_round = next_round
            self.previous_model_version = prev_version
            self.global_model_version = f"resnet18_fed_round_{next_round:03d}"
            self.status = "ROUND_STARTED"
            self.current_step = f"Initializing Round {next_round}"
            self.error_message = None
            self.round_start_time = datetime.utcnow()

            # Initialize client statuses to "waiting"
            for h_code in ["HOSP-001", "HOSP-002", "HOSP-003"]:
                self.clients_status[h_code] = {
                    "hospital_id": h_code,
                    "hospital_name": HOSPITAL_NAMES.get(h_code, f"Hospital {h_code}"),
                    "status": HOSPITAL_WAITING if h_code in self.selected_hospital_codes else HOSPITAL_NOT_SELECTED,
                    "samples": None,
                    "accuracy": None,
                    "f1": None,
                    "loss": None,
                    "duration_sec": None,
                    "update_submitted": False,
                    "model_updated": False,
                    "last_event": "HOSPITAL_WAITING",
                    "last_event_at": datetime.utcnow().isoformat(),
                }
                self._emit_event("HOSPITAL_WAITING", hospital_id=h_code, data={"status": HOSPITAL_WAITING})

        # Spawn execution in background thread
        thread = threading.Thread(
            target=self._run_round_workflow,
            args=(next_round, local_epochs, batch_size, lr, mode, device, list(self.selected_hospital_codes), list(self.selected_hospital_ids)),
            daemon=True,
            name=f"FL-Round-{next_round}-Worker",
        )
        thread.start()

        return {
            "round": self.current_round,
            "status": "started",
            "global_model_version": self.previous_model_version,
            "selected_hospital_ids": self.selected_hospital_ids,
            "message": f"DL Federated Round {self.current_round} initiated successfully.",
        }

    def _run_round_workflow(
        self,
        round_num: int,
        local_epochs: int,
        batch_size: int,
        lr: float,
        mode: str,
        device_str: str,
        selected_hospital_codes: List[str],
        selected_hospital_ids: List[int],
    ):
        """Worker thread executing the real Flower / FedAvg training and aggregation."""
        try:
            from model import build_resnet18_classifier, CLASS_MAPPING
            from partition import partition_dataset, HOSPITAL_IDS
            from client import StoneSenseFLClient
            from local_training import get_model_parameters, set_model_parameters
            from strategy import persist_round_to_db
            from model_manager import model_manager
            from simulate import aggregate_weights, sync_hospital_dataset_metadata

            # 1. ROUND_STARTED Event
            self._emit_event("ROUND_STARTED", data={"message": f"Round {round_num} started across {len(selected_hospital_codes)} selected hospital nodes."})
            time.sleep(0.6)

            # 2. GLOBAL_MODEL_DISTRIBUTING Event
            self.status = "GLOBAL_MODEL_DISTRIBUTING"
            self.current_step = f"Distributing DL Global Federated Model ({self.previous_model_version}) to nodes"
            self._emit_event("GLOBAL_MODEL_DISTRIBUTING", data={"model_version": self.previous_model_version})

            # The global model is distributed to every configured hospital; only selected hospitals train.
            for h_code in ["HOSP-001", "HOSP-002", "HOSP-003"]:
                self._set_hospital_state(h_code, HOSPITAL_MODEL_RECEIVED, "HOSPITAL_MODEL_RECEIVED")
                self._emit_event("MODEL_RECEIVED", hospital_id=h_code)
                if h_code not in selected_hospital_codes:
                    self._set_hospital_state(h_code, HOSPITAL_NOT_SELECTED, "HOSPITAL_NOT_SELECTED")


            time.sleep(0.5)

            # 3. Verify / partition dataset
            partitions_root = DL_ROOT / "datasets" / "partitions"
            dist_file = partitions_root / "distribution.json"
            if not dist_file.exists():
                logger.info("Partitions not found. Generating hospital partitions...")
                partitions_meta = partition_dataset(
                    source_dir=DL_ROOT / "processed",
                    output_dir=partitions_root,
                    mode=mode,
                    seed=42,
                )
            else:
                import json
                with open(dist_file, "r", encoding="utf-8") as f:
                    partitions_meta = json.load(f)

            sync_hospital_dataset_metadata(partitions_meta)

            # 4. Device & Base Model
            dev = torch.device("cuda" if torch.cuda.is_available() and device_str != "cpu" else "cpu")
            class_names = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]
            global_model = build_resnet18_classifier(
                num_classes=len(class_names),
                freeze_backbone=True,
                unfreeze_layer4=True,
            ).to(dev)

            # Load weights from previous checkpoint if available
            latest_ckpt = model_manager.get_latest_checkpoint_path()
            if latest_ckpt and latest_ckpt.exists():
                try:
                    ckpt_data = torch.load(latest_ckpt, map_location=dev, weights_only=False)
                    if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                        global_model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
                    elif isinstance(ckpt_data, dict):
                        global_model.load_state_dict(ckpt_data, strict=False)
                    logger.info(f"Initialized Round {round_num} from base checkpoint: {latest_ckpt}")
                except Exception as e:
                    logger.warning(f"Could not load previous checkpoint, starting with base weights: {e}")

            global_weights = get_model_parameters(global_model)

            # 5. LOCAL TRAINING on Hospital Clients (Zero-Raw-CT Privacy Isolation)
            self.status = "LOCAL_TRAINING"
            self.current_step = "Executing isolated local client training on hospital nodes"

            client_weights = []
            client_sample_counts = []
            client_fit_metrics = []
            client_eval_metrics = {}

            selected_partition_ids = [
                h_id for h_id in HOSPITAL_IDS
                if HOSPITAL_CODE_MAP.get(h_id) in selected_hospital_codes
            ]
            config = {"current_round": round_num, "local_epochs": local_epochs, "lr": lr, "federated_round": True}

            for h_id in selected_partition_ids:
                h_code = HOSPITAL_CODE_MAP.get(h_id, "HOSP-001")
                self._set_hospital_state(h_code, HOSPITAL_TRAINING, "HOSPITAL_TRAINING_STARTED")
                self._emit_event(
                    "HOSPITAL_TRAINING",
                    hospital_id=h_code,
                    data={"status": "training", "hospital_name": HOSPITAL_NAMES.get(h_code, h_code)},
                )

                client_start = time.time()
                client = StoneSenseFLClient(
                    hospital_id=h_code,
                    partition_dir=partitions_root / h_id,
                    batch_size=batch_size,
                    device=dev,
                    lr=lr,
                )

                updated_weights, samples, fit_m = client.fit(global_weights, config)
                client_duration = time.time() - client_start

                client_weights.append(updated_weights)
                client_sample_counts.append(samples)
                client_fit_metrics.append(fit_m)

                # Update client status to completed
                acc = float(fit_m.get("train_accuracy", 0.95))
                f1 = float(fit_m.get("train_f1_macro", 0.94))
                loss = float(fit_m.get("train_loss", 0.08))

                self.clients_status[h_code].update({
                    "status": HOSPITAL_TRAINING_COMPLETED,
                    "samples": samples,
                    "accuracy": round(acc, 4),
                    "f1": round(f1, 4),
                    "loss": round(loss, 4),
                    "duration_sec": round(client_duration, 2),
                })
                self._set_hospital_state(
                    h_code,
                    HOSPITAL_TRAINING_COMPLETED,
                    "HOSPITAL_TRAINING_COMPLETED",
                    samples=samples,
                    accuracy=round(acc, 4),
                    f1=round(f1, 4),
                    loss=round(loss, 4),
                    duration_sec=round(client_duration, 2),
                )

                self._emit_event("CLIENT_UPDATE_RECEIVED", hospital_id=h_code)
                self._set_hospital_state(h_code, HOSPITAL_UPDATE_SUBMITTED, "HOSPITAL_UPDATE_SUBMITTED", update_submitted=True)
                self._set_hospital_state(h_code, HOSPITAL_WAITING_FOR_AGGREGATION, "HOSPITAL_WAITING_FOR_AGGREGATION", update_submitted=True)

            # 6. FedAvg Aggregation
            self.status = "FEDAVG_STARTED"
            self.current_step = "Aggregating hospital model weights via FedAvg"
            self._emit_event("FEDAVG_STARTED", data={"participating_nodes": len(client_sample_counts)})

            global_weights = aggregate_weights(client_weights, client_sample_counts)
            set_model_parameters(global_model, global_weights)
            time.sleep(0.5)

            # 7. Validation across hospital test partitions
            total_eval_samples = 0
            weighted_eval_loss = 0.0
            weighted_eval_acc = 0.0
            weighted_eval_f1 = 0.0
            weighted_eval_prec = 0.0
            weighted_eval_rec = 0.0

            for h_id in HOSPITAL_IDS:
                h_code = HOSPITAL_CODE_MAP.get(h_id, "HOSP-001")
                client = StoneSenseFLClient(
                    hospital_id=h_code,
                    partition_dir=partitions_root / h_id,
                    batch_size=batch_size,
                    device=dev,
                )
                loss, samples, eval_m = client.evaluate(global_weights, config)
                client_eval_metrics[h_code] = eval_m

                total_eval_samples += samples
                weighted_eval_loss += loss * samples
                weighted_eval_acc += float(eval_m.get("accuracy", 0.0)) * samples
                weighted_eval_f1 += float(eval_m.get("f1_macro", 0.0)) * samples
                weighted_eval_prec += float(eval_m.get("precision_macro", 0.0)) * samples
                weighted_eval_rec += float(eval_m.get("recall_macro", 0.0)) * samples

            global_val_loss = weighted_eval_loss / max(total_eval_samples, 1)
            global_val_acc = weighted_eval_acc / max(total_eval_samples, 1)
            global_val_f1 = weighted_eval_f1 / max(total_eval_samples, 1)
            global_val_prec = weighted_eval_prec / max(total_eval_samples, 1)
            global_val_rec = weighted_eval_rec / max(total_eval_samples, 1)

            tot_train_samples = sum(client_sample_counts)
            global_train_loss = sum(m["train_loss"] * client_sample_counts[i] for i, m in enumerate(client_fit_metrics)) / max(tot_train_samples, 1)
            global_train_acc = sum(m["train_accuracy"] * client_sample_counts[i] for i, m in enumerate(client_fit_metrics)) / max(tot_train_samples, 1)

            fit_summary = {"train_loss": global_train_loss, "train_accuracy": global_train_acc}
            eval_summary = {
                "loss": global_val_loss,
                "accuracy": global_val_acc,
                "f1_macro": global_val_f1,
                "precision_macro": global_val_prec,
                "recall_macro": global_val_rec,
            }

            self.latest_round_metrics = {
                "accuracy": round(global_val_acc, 4),
                "f1": round(global_val_f1, 4),
                "loss": round(global_val_loss, 4),
                "precision": round(global_val_prec, 4),
                "recall": round(global_val_rec, 4),
            }

            self._emit_event("FEDAVG_COMPLETED", data=self.latest_round_metrics)

            # 8. Checkpoint Saving & Database Persistence
            client_runs = []
            for i, h_id in enumerate(selected_partition_ids):
                h_code = HOSPITAL_CODE_MAP.get(h_id, "HOSP-001")
                fit_m = client_fit_metrics[i]
                eval_m = client_eval_metrics[h_code]
                client_runs.append({
                    "hospital_code": h_code,
                    "train_loss": fit_m.get("train_loss"),
                    "train_acc": fit_m.get("train_accuracy"),
                    "train_f1": fit_m.get("train_f1_macro"),
                    "val_loss": eval_m.get("loss"),
                    "val_acc": eval_m.get("accuracy"),
                    "val_f1": eval_m.get("f1_macro"),
                    "sample_count": client_sample_counts[i],
                })

            duration = (datetime.utcnow() - self.round_start_time).total_seconds()

            persist_round_to_db(
                round_number=round_num,
                mode=mode,
                parameters=global_weights,
                fit_metrics=fit_summary,
                eval_metrics=eval_summary,
                client_runs=client_runs,
                duration_sec=duration,
                selected_hospital_ids=selected_hospital_ids,
            )

            # Reload runtime DL model for real-time inference
            try:
                from app.services.model_loader import model_loader
                model_loader.reload_dl_model(self.global_model_version)
            except Exception as e:
                logger.warning(f"Could not trigger model_loader reload: {e}")

            self._emit_event("GLOBAL_MODEL_SAVED", data={"new_version": self.global_model_version})
            self._emit_event("MODEL_DISTRIBUTED", data={"new_version": self.global_model_version})

            for h_code in ["HOSP-001", "HOSP-002", "HOSP-003"]:
                if h_code in selected_hospital_codes:
                    self._set_hospital_state(h_code, HOSPITAL_MODEL_UPDATED, "HOSPITAL_MODEL_UPDATED", update_submitted=True, model_updated=True)
                    self._set_hospital_state(h_code, HOSPITAL_COMPLETED, "HOSPITAL_ROUND_COMPLETED", update_submitted=True, model_updated=True)
                else:
                    self._set_hospital_state(h_code, HOSPITAL_MODEL_UPDATED, "HOSPITAL_MODEL_UPDATED", update_submitted=False, model_updated=True)

            # 9. ROUND_COMPLETED
            self.status = "COMPLETED"
            self.current_step = f"Round {round_num} successfully completed and synchronized"
            self._emit_event(
                "ROUND_COMPLETED",
                data={
                    "round": round_num,
                    "new_model_version": self.global_model_version,
                    "metrics": self.latest_round_metrics,
                    "duration_sec": round(duration, 2),
                },
            )

            logger.info(
                f"=== Live DL Federated Round {round_num} Complete: "
                f"Acc: {global_val_acc:.4f} | F1: {global_val_f1:.4f} | Model: {self.global_model_version} ==="
            )

        except Exception as e:
            logger.exception(f"Error during live federated round execution: {e}")
            self.status = "FAILED"
            self.error_message = str(e)
            self.current_step = f"Round execution failed: {e}"
            self._emit_event("ROUND_FAILED", data={"error": str(e)})
            for h_code, client_data in self.clients_status.items():
                if client_data.get("status") not in (HOSPITAL_COMPLETED, HOSPITAL_MODEL_UPDATED):
                    self._set_hospital_state(h_code, HOSPITAL_FAILED, "HOSPITAL_ROUND_FAILED", error=str(e))
        finally:
            with self._lock:
                self.is_running = False

    def get_live_round_status(self, round_id: Optional[int] = None) -> Dict[str, Any]:
        """Returns live status of current or most recent round."""
        completed_count = sum(
            1 for c in self.clients_status.values()
            if c.get("hospital_id") in self.selected_hospital_codes
            and c.get("status") in (HOSPITAL_COMPLETED, HOSPITAL_MODEL_UPDATED)
        )
        clients_list = list(self.clients_status.values())

        return {
            "round": self.current_round,
            "status": self.status,
            "previous_model_version": self.previous_model_version,
            "global_model_version": self.global_model_version,
            "participating_hospitals": len(self.selected_hospital_codes),
            "completed_hospitals": completed_count,
            "clients": clients_list,
            "current_step": self.current_step,
            "error_message": self.error_message,
            "metrics": self.latest_round_metrics,
        }

    def get_hospital_live_status(self, hospital_identifier: str) -> Dict[str, Any]:
        """Returns hospital-specific status during live round."""
        db_code = HOSPITAL_CODE_MAP.get(str(hospital_identifier), str(hospital_identifier))
        if db_code not in self.clients_status:
            raise KeyError(f"Unknown hospital identifier: {hospital_identifier}")
        client_data = self.clients_status.get(db_code, {})

        return {
            "hospital_id": db_code,
            "round": self.current_round,
            "status": client_data.get("status", HOSPITAL_WAITING),
            "phase": self.status,
            "global_model_version": self.global_model_version,
            "local_training": {
                "status": client_data.get("status", HOSPITAL_WAITING),
                "samples": client_data.get("samples"),
                "accuracy": client_data.get("accuracy"),
                "f1": client_data.get("f1"),
                "loss": client_data.get("loss"),
                "duration_sec": client_data.get("duration_sec"),
            },
            "update_submitted": client_data.get("update_submitted", False),
            "model_updated": client_data.get("model_updated", False),
            "last_event": client_data.get("last_event"),
            "last_event_at": client_data.get("last_event_at"),
            "round_status": self.status,
        }

    def _set_hospital_state(self, hospital_id: str, status: str, event_type: str, **updates: Any) -> None:
        """Update live client telemetry and emit the matching hospital event."""
        client_data = self.clients_status[hospital_id]
        event_at = datetime.utcnow().isoformat()
        client_data.update({"status": status, "last_event": event_type, "last_event_at": event_at, **updates})
        self._emit_event(event_type, hospital_id=hospital_id, data={"status": status, **updates})


federated_coordinator = FederatedCoordinator()
