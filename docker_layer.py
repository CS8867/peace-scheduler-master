import docker
import logging
import time
from typing import Dict, List

# Initialize the Docker client (communicates with the Docker Daemon on your machine)
client = docker.from_env()

class DockerLayer:
    
    @staticmethod
    def start_container(
        image: str, 
        name: str, 
        command: str, 
        gpu_idx: int, 
        mps_percentage: int, 
        volumes: Dict[str, Dict[str, str]] = {},
        envs: Dict[str, str] = {},
        interactive: bool = False
    ) -> str:
        """
        Starts a container with specific MPS configuration and mounted volumes.
        Returns the container ID (short version).

        If interactive=True, the container is started with stdin_open and tty
        (equivalent to 'docker run -it'), which keeps shells like 'bash' alive
        in detached mode so that commands can be exec'd into it later.
        """
        try:
            # 1. Setup the environment variables. The run command looks at all variables passed via the environment dict. The Dockerfile equivalent of this is the ENV instruction, which sets environment variables for the container at runtime.
            environment = envs.copy()
            environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_percentage)
            environment["NVIDIA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            # 2. Configure Device Requests (Standard way to request GPUs in Python Docker SDK)
            # Note: For MPS, we usually just expose the specific GPU ID.
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]

            logging.info(f"Spawning {name} on GPU {gpu_idx} with {mps_percentage}% MPS...")

            # 3. Run the Container
            container = client.containers.run(
                image=image,
                command=command,
                name=name,
                detach=True,                 # Run in background
                environment=environment,
                volumes=volumes,             # Mount shared checkpoint folders
                device_requests=device_requests,
                ipc_mode="host",             # CRITICAL for PyTorch/MPS shared memory
                cap_add=["SYS_ADMIN"],       # Often needed for GPU profiling/management tools
                auto_remove=False,           # We want to inspect it after it stops, so don't auto-delete
                stdin_open=interactive,       # -i flag: keep STDIN open
                tty=interactive              # -t flag: allocate a pseudo-TTY
            )
            
            return container.short_id

        except Exception as e:
            logging.error(f"Failed to start container {name}: {str(e)}")
            raise e

    @staticmethod
    def exec_command(
        container_id: str,
        command: str,
        workdir: str = None,
        detach: bool = False
    ):
        """
        Executes an arbitrary command inside a running container.
        This is the equivalent of 'docker exec [-d] [-w workdir] <container> <cmd>'.

        Args:
            container_id: ID or name of the running container.
            command:      The shell command string to execute.
            workdir:      Optional working directory inside the container.
            detach:       If True, run in the background and return immediately.
                          If False (default), block until the command finishes and
                          return (exit_code, output).

        Returns:
            If detach=False: tuple (exit_code: int, output: str)
            If detach=True:  None (command is fire-and-forget)
        """
        try:
            container = client.containers.get(container_id)
            logging.info(
                f"Exec inside {container_id}: {command}"
                + (f"  (workdir={workdir})" if workdir else "")
            )

            exit_code, output = container.exec_run(
                command,
                workdir=workdir,
                detach=detach,
            )

            if detach:
                logging.info(f"Command launched (detached) in {container_id}.")
                return None

            decoded = output.decode("utf-8") if output else ""
            if exit_code == 0:
                logging.info(f"Command succeeded in {container_id}. Output:\n{decoded}")
            else:
                logging.error(f"Command failed (exit {exit_code}) in {container_id}. Output:\n{decoded}")

            return exit_code, decoded

        except Exception as e:
            logging.error(f"Error executing command in {container_id}: {str(e)}")
            raise e

    @staticmethod
    def trigger_checkpoint(container_id: str, checkpoint_cmd: str) -> bool:
        """
        Executes a command INSIDE the running container to force a save.
        Example cmd: "python save_model.py" or a custom signal handler script.
        """
        try:
            container = client.containers.get(container_id)
            logging.info(f"Executing checkpoint command inside {container_id}...")
            
            # exec_run executes the command inside the container
            exit_code, output = container.exec_run(checkpoint_cmd, detach=False)
            
            if exit_code == 0:
                logging.info(f"Checkpoint successful for {container_id}")
                return True
            else:
                logging.error(f"Checkpoint failed for {container_id}. Output: {output.decode('utf-8')}")
                return False
        except Exception as e:
            logging.error(f"Error triggering checkpoint on {container_id}: {str(e)}")
            return False

    @staticmethod
    def stop_and_remove(container_id: str):
        """
        Hard stop and removal of a container.
        """
        try:
            container = client.containers.get(container_id)
            container.stop(timeout=5) # Give it 5 seconds to wrap up naturally
            # container.remove(force=True)
            logging.info(f"Container {container_id} stopped and removed.")
        except docker.errors.NotFound:
            logging.warning(f"Container {container_id} not found (already gone?).")
        except Exception as e:
            logging.error(f"Error stopping container {container_id}: {str(e)}")

    @staticmethod
    def is_container_running(container_id: str) -> bool:
        """
        Checks if a container is currently running.
        """
        try:
            container = client.containers.get(container_id)
            return container.status == 'running'
        except docker.errors.NotFound:
            return False
        
    @staticmethod
    def send_signal(container_id: str, signal_name: str = "SIGUSR1"):
        """
        Sends a specific Linux signal to the container.
        """
        try:
            container = client.containers.get(container_id)
            logging.info(f"Sending signal {signal_name} to {container_id}...")
            container.kill(signal=signal_name) # 'kill' is the Docker SDK name for sending signals
            return True
        except Exception as e:
            logging.error(f"Failed to send signal to {container_id}: {str(e)}")
            return False

    @staticmethod
    def wait_for_exit(container_id: str):
        """
        BLOCKING call. logic pauses here until this container dies.
        Used for Step 2: "Wait for Job 1 to complete".
        """
        try:
            container = client.containers.get(container_id)
            container.wait() # This blocks the thread
            logging.info(f"Container {container_id} has exited.")
        except Exception as e:
            logging.error(f"Error waiting for container {container_id}: {str(e)}")

    @staticmethod
    def wait_for_log_message(container_id: str):
        """
        Checks to see if the container produced the first log message. This will make sure that the job is actually ready to run, before we kill the old container.
        """

        

        # Need to continuously monitor for the Ready message

        while True:
            container = client.containers.get(container_id)
            logs = container.logs()
            if "Ready" in logs.decode("utf-8"):  # Do we have to do utf-8?
                logging.info(f"Container {container_id} is actually in ready state")
                break
            time.sleep(1)


    @staticmethod
    def get_host_pids(container_id: str) -> List[int]:
        """
        Returns ALL host PIDs of processes running inside the container.
        Uses container.top() so we capture child processes (e.g. python
        spawned by bash), not just PID 1.
        """
        try:
            container = client.containers.get(container_id)
            top_output = container.top()
            # top() returns {'Titles': [...], 'Processes': [[pid, ...], ...]}
            # The PID column is the second column (index 1) by default.
            pid_index = top_output['Titles'].index('PID')
            pids = [int(proc[pid_index]) for proc in top_output['Processes']]
            return pids
        except Exception as e:
            logging.error(f"Error getting host PIDs for {container_id}: {str(e)}")
            return []

    @staticmethod
    def container_logs_contain(container_id: str, expected_text: str, tail: int = 100) -> bool:
        """
        Returns True if the container's recent logs contain the expected text.
        """
        try:
            container = client.containers.get(container_id)
            logs = container.logs(tail=tail).decode("utf-8", errors="replace")
            return expected_text in logs
        except Exception as e:
            logging.error(f"Error checking logs for {container_id}: {str(e)}")
            return False

    @staticmethod
    def has_inference_output(container_id: str) -> bool:
        """
        Returns True if the container has produced at least one inference
        prediction line in its logs (e.g. 'predictions:').
        This is used as a container-level readiness signal by the Router
        so the inference script itself does not need modification.
        """
        return DockerLayer.container_logs_contain(container_id, "predictions:", tail=50)