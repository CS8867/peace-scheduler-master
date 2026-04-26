import time
import logging
from dataclasses import dataclass
from typing import List, Optional
from docker_layer import DockerLayer
import subprocess


@dataclass
class PeaceRunningJob:
    container_id: str
    container_name: str
    job_name: str
    status: str
    gpu_idx: Optional[int]
    mps_percentage: Optional[int]


@dataclass
class PeaceNodeState:
    running_jobs: List[PeaceRunningJob]

    @property
    def running_count(self) -> int:
        return len(self.running_jobs)

class Monitor:
    @staticmethod
    def get_peace_node_state(name_prefix: str = "peace-") -> PeaceNodeState:
        """
        Returns the currently running PEACE-managed containers on the node.
        """
        containers = DockerLayer.list_containers(all_containers=False, name_prefix=name_prefix)
        running_jobs = []

        for container in containers:
            gpu_idx = container.get("gpu_idx")
            mps_percentage = container.get("mps_percentage")
            running_jobs.append(
                PeaceRunningJob(
                    container_id=str(container["id"]),
                    container_name=str(container["name"]),
                    job_name=str(container["name"])[len(name_prefix):],
                    status=str(container["status"]),
                    gpu_idx=int(gpu_idx) if gpu_idx is not None and str(gpu_idx).isdigit() else None,
                    mps_percentage=int(mps_percentage) if mps_percentage is not None and str(mps_percentage).isdigit() else None,
                )
            )

        logging.info(
            "Monitor: PEACE node state -> %s running job(s): %s",
            len(running_jobs),
            [job.container_name for job in running_jobs],
        )
        return PeaceNodeState(running_jobs=running_jobs)

    @staticmethod
    def get_peace_running_job_count(name_prefix: str = "peace-") -> int:
        """
        Returns the number of running PEACE-managed containers on the node.
        """
        return Monitor.get_peace_node_state(name_prefix=name_prefix).running_count

    @staticmethod
    def wait_for_any_exit(container_ids: List[str], poll_interval: int = 2) -> str:
        """
        Watches a LIST of containers. 
        Returns the ID of the FIRST one that finishes.
        """
        logging.info(f"Monitor: Watching {len(container_ids)} containers: {container_ids}")
        
        while True:
            for c_id in container_ids:
                if not DockerLayer.is_container_running(c_id):
                    logging.info(f"Monitor: Alert! Container {c_id} has finished.")
                    return c_id
            
            time.sleep(poll_interval)


    @staticmethod
    def wait_for_gpu_run(container_id: str, poll_interval: int = 2, timeout: int = 120) -> str:
        """
        Waits until the container's workload is actively using the GPU.
        Returns the container_id on success, or None on timeout / early exit.
        """
        # First, wait for the container to be in "Running" state
        while True:
            if DockerLayer.is_container_running(container_id):
                logging.info(f"Monitor: Container {container_id} is running. Now checking for GPU usage...")
                break
            time.sleep(poll_interval)

        start_time = time.time()

        # Poll nvidia-smi until any process from this container appears on the GPU
        while True:
            # Escape hatch: container died while we were waiting
            if not DockerLayer.is_container_running(container_id):
                logging.error(f"Monitor: Container {container_id} exited before using the GPU.")
                return None

            # Escape hatch: timeout
            if time.time() - start_time > timeout:
                logging.error(f"Monitor: Timed out waiting for {container_id} to use the GPU.")
                return None

            # Fetch ALL host PIDs belonging to this container (handles bash -> python case)
            host_pids = DockerLayer.get_host_pids(container_id)

            # Query nvidia-smi for PIDs currently using GPU compute
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            gpu_pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]

            # Check if ANY container PID is in the GPU process list
            for pid in host_pids:
                if str(pid) in gpu_pids:
                    logging.info(f"Monitor: Container {container_id} (PID {pid}) is using the GPU.")
                    return container_id

            time.sleep(poll_interval)

    @staticmethod
    def wait_for_log_message(
        container_id: str,
        expected_text: str,
        poll_interval: float = 0.2,
        timeout: int = 120,
        tail: int = 100,
    ) -> str:
        """
        Waits until the container emits a specific log marker.
        Returns the container_id on success, or None on timeout / early exit.
        """
        logging.info(f"Monitor: Waiting for log marker '{expected_text}' from container {container_id}.")
        start_time = time.time()

        while True:
            if DockerLayer.container_logs_contain(container_id, expected_text, tail=tail):
                logging.info(f"Monitor: Container {container_id} emitted marker '{expected_text}'.")
                return container_id

            if not DockerLayer.is_container_running(container_id):
                logging.error(f"Monitor: Container {container_id} exited before emitting '{expected_text}'.")
                return None

            if time.time() - start_time > timeout:
                logging.error(f"Monitor: Timed out waiting for {container_id} to emit '{expected_text}'.")
                return None

            time.sleep(poll_interval)