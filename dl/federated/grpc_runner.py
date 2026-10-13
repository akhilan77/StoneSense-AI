"""StoneSense-AI Flower gRPC Process Runner & Manager.

Provides process-level orchestration for real Flower gRPC federated learning:
- Binds server to 127.0.0.1 on an available test/runtime port.
- Spawns independent server and hospital client subprocesses.
- Generates isolated deterministic synthetic training partitions.
- Manages health checks, process lifecycle, timeouts, stdout/stderr capture, and guaranteed cleanup.
"""

from pathlib import Path
import sys
import os
import time
import socket
import logging
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DL_ROOT = PROJECT_ROOT / "dl"

logger = logging.getLogger("StoneSenseGRPCRunner")


def find_free_port() -> int:
    """Finds an available TCP port bound to 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 15.0) -> bool:
    """Waits until a TCP port is open and accepting socket connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect((host, port))
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.2)
    return False


def create_synthetic_partitions(
    base_dir: Path,
    hospital_ids: List[str] = ["hospital_1", "hospital_2"],
    images_per_class: int = 2,
    image_size: Tuple[int, int] = (64, 64),
) -> Path:
    """Generates isolated deterministic synthetic dataset partitions for fast multi-process FL testing."""
    classes = ["Cyst", "Normal", "Stone", "Tumor"]
    colors = {
        "Cyst": (100, 150, 200),
        "Normal": (80, 180, 80),
        "Stone": (220, 120, 80),
        "Tumor": (200, 80, 120),
    }

    distribution_meta: Dict[str, Any] = {"distributions": {}, "hospital_names": {}}

    for h_idx, h_id in enumerate(hospital_ids):
        h_dir = base_dir / h_id
        distribution_meta["distributions"][h_id] = {"train": {}, "validation": {}}
        distribution_meta["hospital_names"][h_id] = f"Synthetic Hospital {h_id}"

        for split in ["train", "validation", "test"]:
            for c_idx, cname in enumerate(classes):
                cdir = h_dir / split / cname
                cdir.mkdir(parents=True, exist_ok=True)

                distribution_meta["distributions"][h_id].setdefault(split, {})[cname] = images_per_class

                for img_i in range(images_per_class):
                    # Deterministic color variation per hospital, class, and image index
                    r = (colors[cname][0] + (h_idx * 20) + (img_i * 10)) % 256
                    g = (colors[cname][1] + (h_idx * 15) + (img_i * 10)) % 256
                    b = (colors[cname][2] + (h_idx * 10) + (img_i * 10)) % 256

                    img = Image.new("RGB", image_size, color=(r, g, b))
                    img_path = cdir / f"syn_{h_id}_{split}_{cname}_{img_i}.png"
                    img.save(img_path)

    # Write distribution.json
    dist_file = base_dir / "distribution.json"
    with open(dist_file, "w", encoding="utf-8") as f:
        json.dump(distribution_meta, f, indent=2)

    return base_dir


class FlowerGRPCRunner:
    """Process manager for launching and tracking Flower gRPC server and client subprocesses."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        rounds: int = 1,
        min_clients: int = 2,
        models_dir: Optional[Path] = None,
        partitions_dir: Optional[Path] = None,
        db_persist: bool = False,
        python_executable: Optional[str] = None,
    ):
        self.host = host
        self.port = port or find_free_port()
        self.server_address = f"{self.host}:{self.port}"
        self.rounds = rounds
        self.min_clients = min_clients
        self.models_dir = Path(models_dir) if models_dir else (DL_ROOT / "models")
        self.partitions_dir = Path(partitions_dir) if partitions_dir else (DL_ROOT / "datasets" / "partitions")
        self.db_persist = db_persist
        self.python_executable = python_executable or sys.executable

        self.server_process: Optional[subprocess.Popen] = None
        self.client_processes: Dict[str, subprocess.Popen] = {}
        self.server_logs: Tuple[str, str] = ("", "")
        self.client_logs: Dict[str, Tuple[str, str]] = {}
        self.exit_codes: Dict[str, int] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def start_server(self) -> subprocess.Popen:
        """Launches the Flower gRPC server in a separate process."""
        cmd = [
            self.python_executable,
            str(DL_ROOT / "federated" / "server.py"),
            "--address", self.server_address,
            "--rounds", str(self.rounds),
            "--min-clients", str(self.min_clients),
            "--models-dir", str(self.models_dir),
        ]
        if not self.db_persist:
            cmd.append("--no-db")

        env = os.environ.copy()
        env["STONESENSE_MODELS_DIR"] = str(self.models_dir)
        env["PYTHONUNBUFFERED"] = "1"
        env["FLWR_TELEMETRY_ENABLED"] = "0"

        logger.info(f"Starting Flower server process on {self.server_address}...")
        self.server_log_path = self.models_dir / "server.log"
        self._server_file = open(self.server_log_path, "w+", encoding="utf-8")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=self._server_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return self.server_process

    def start_client(
        self,
        hospital_id: str,
        max_batches: int = 1,
        batch_size: int = 4,
        lr: float = 0.001,
    ) -> subprocess.Popen:
        """Launches an independent hospital client process connecting over gRPC."""
        cmd = [
            self.python_executable,
            str(DL_ROOT / "federated" / "client.py"),
            "--hospital", hospital_id,
            "--server", self.server_address,
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--partitions-dir", str(self.partitions_dir),
            "--max-batches", str(max_batches),
        ]

        env = os.environ.copy()
        env["STONESENSE_MODELS_DIR"] = str(self.models_dir)
        env["PYTHONUNBUFFERED"] = "1"
        env["FLWR_TELEMETRY_ENABLED"] = "0"

        logger.info(f"Starting Flower client process [{hospital_id}] connecting to {self.server_address}...")
        client_log_path = self.models_dir / f"{hospital_id}.log"
        if not hasattr(self, "_client_files"):
            self._client_files = {}
        client_file = open(client_log_path, "w+", encoding="utf-8")
        self._client_files[hospital_id] = client_file

        proc = subprocess.Popen(
            cmd,
            stdout=client_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self.client_processes[hospital_id] = proc
        return proc

    def run_round(
        self,
        client_ids: List[str] = ["hospital_1", "hospital_2"],
        max_batches: int = 1,
        batch_size: int = 4,
        server_startup_timeout: float = 15.0,
        round_timeout: float = 90.0,
    ) -> Dict[str, Any]:
        """Executes a complete multi-process Flower gRPC round with guaranteed cleanup."""
        try:
            # 1. Start server process
            self.start_server()

            # 2. Wait for server socket to bind and accept connections
            if not wait_for_port(self.port, host=self.host, timeout=server_startup_timeout):
                # Close server file and read
                self._server_file.flush()
                self._server_file.seek(0)
                err = self._server_file.read()
                raise TimeoutError(
                    f"Flower gRPC server failed to listen on {self.server_address} within {server_startup_timeout}s. "
                    f"Server log:\n{err}"
                )

            # 3. Launch enrolled client processes
            for h_id in client_ids:
                self.start_client(hospital_id=h_id, max_batches=max_batches, batch_size=batch_size)

            # 4. Wait for server and client completion with proactive error detection
            start_time = time.time()
            server_finished = False

            while time.time() - start_time < round_timeout:
                # Check if server finished
                if self.server_process.poll() is not None:
                    server_finished = True
                    break

                # Check if any client failed prematurely
                for h_id, c_proc in self.client_processes.items():
                    if c_proc.poll() is not None and c_proc.poll() != 0:
                        c_file = self._client_files[h_id]
                        c_file.flush()
                        c_file.seek(0)
                        c_log = c_file.read()
                        self._server_file.flush()
                        self._server_file.seek(0)
                        s_log = self._server_file.read()
                        raise RuntimeError(
                            f"Flower client [{h_id}] exited prematurely with code {c_proc.returncode}.\n"
                            f"Client Log:\n{c_log}\nServer Log:\n{s_log}"
                        )

                time.sleep(0.5)

            if not server_finished:
                self._server_file.flush()
                self._server_file.seek(0)
                s_log = self._server_file.read()
                raise TimeoutError(
                    f"Flower gRPC round timed out after {round_timeout}s.\n"
                    f"Server Log:\n{s_log}"
                )

            # Wait briefly for client processes to finish cleanly
            for h_id, c_proc in self.client_processes.items():
                try:
                    c_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    c_proc.terminate()

            # Read server log
            self._server_file.flush()
            self._server_file.seek(0)
            server_out = self._server_file.read()
            self.server_logs = (server_out, "")
            self.exit_codes["server"] = self.server_process.returncode

            # Read client logs
            for h_id, c_proc in self.client_processes.items():
                c_file = self._client_files[h_id]
                c_file.flush()
                c_file.seek(0)
                c_out = c_file.read()
                self.client_logs[h_id] = (c_out, "")
                self.exit_codes[h_id] = c_proc.returncode

            success = (
                self.exit_codes.get("server") == 0
                and all(code == 0 for k, code in self.exit_codes.items() if k != "server")
            )

            expected_ckpt = self.models_dir / "federated" / f"resnet18_fed_round_{self.rounds:03d}.pth"

            return {
                "success": success,
                "server_address": self.server_address,
                "rounds": self.rounds,
                "client_ids": client_ids,
                "exit_codes": self.exit_codes,
                "server_logs": self.server_logs,
                "client_logs": self.client_logs,
                "checkpoint_path": expected_ckpt if expected_ckpt.exists() else None,
            }

        finally:
            self.cleanup()

    def cleanup(self):
        """Terminates and reaps all child processes unconditionally."""
        all_procs = list(self.client_processes.values())
        if self.server_process is not None:
            all_procs.append(self.server_process)

        for proc in all_procs:
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.5)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass

        if hasattr(self, "_server_file") and self._server_file and not self._server_file.closed:
            try:
                self._server_file.close()
            except Exception:
                pass
        if hasattr(self, "_client_files"):
            for f in self._client_files.values():
                if f and not f.closed:
                    try:
                        f.close()
                    except Exception:
                        pass
