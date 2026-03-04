import time
import logging
from typing import List
from docker_layer import DockerLayer
import subprocess

class Monitor:
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