import time
import logging
from docker_layer import DockerLayer


class Router:
    """
    Container-level load balancer/router.

    Instead of HTTP health checks, readiness is determined by inspecting
    container status and log output through the Docker API.  This means
    the inference script does NOT need any modification.
    """

    def __init__(self):
        self.active_container = None   # container_id of the active backend
        self.logger = logging.getLogger("Router")

    def set_backend(self, container_id):
        self.active_container = container_id
        self.logger.info(f"Router pointing to container {container_id}")

    def switch_backend(self, new_container_id, poll_interval=0.2, timeout=120):
        """
        Switch the active backend from the current container to *new_container_id*.

        Readiness is confirmed when the new container is running AND has
        produced its first inference log line (checked via Docker logs).
        
        Returns the switch duration (downtime) in seconds, or None on timeout.
        """
        old = self.active_container
        self.logger.info(f"Initiating switch: {old} -> {new_container_id}")
        switch_start = time.time()

        while time.time() - switch_start < timeout:
            # Container must be running
            if not DockerLayer.is_container_running(new_container_id):
                time.sleep(poll_interval)
                continue

            # Container must have produced inference output
            if DockerLayer.has_inference_output(new_container_id):
                self.set_backend(new_container_id)
                switch_duration = time.time() - switch_start
                self.logger.info(
                    f"Router switch complete in {switch_duration:.4f}s (downtime)"
                )
                return switch_duration

            time.sleep(poll_interval)

        self.logger.error(f"Router switch timed out after {timeout}s!")
        return None
