from collections import deque
from dataclasses import dataclass, field
import logging
import time
from typing import Deque, Dict, List, Optional

from docker_layer import DockerLayer
from monitor import Monitor, PeaceNodeState


@dataclass
class Job:
    name: str
    job_type: str
    command: str
    gpu_idx: int
    mps_percentage: int
    envs: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    readiness_log_marker: Optional[str] = None


class Scheduler:
    def __init__(
        self,
        image_name: str,
        volumes: Dict[str, Dict[str, str]],
        peace_prefix: str = "peace-",
    ) -> None:
        self.image_name = image_name
        self.volumes = volumes
        self.peace_prefix = peace_prefix
        self.job_queue: Deque[Job] = deque(self._build_hardcoded_jobs())
        self.last_node_state: Optional[PeaceNodeState] = None
        self.active_jobs_by_id: Dict[str, Job] = {}
        self.active_jobs_by_name: Dict[str, Job] = {}
        self.redeploy_generation: Dict[str, int] = {}

    def _build_hardcoded_jobs(self) -> List[Job]:
        train_jobs_dir = "/app/train-jobs"
        serve_jobs_dir = "/app/serve-jobs"
        train_workdir = "/root/mlprofiler/workloads/train"
        serve_workdir = "/root/mlprofiler/workloads/inference"
        checkpoint_path = "/app/checkpoints/recommend_train_ckpt.pt"

        serve_recommend_cmd = (
            f"bash -c 'cd {serve_workdir} && python {serve_jobs_dir}/recommend-inference.py"
            " --batch_size 2"
            " --model_name bert-base-cased"
            " --profile_nstep 10000"
            " --log_dir test'"
        )
        train_recommend_cmd = (
            f"python {train_jobs_dir}/recommend-train.py"
            " --batch_size 2"
            " --model_name bert-large-cased"
            " --profile_nstep 1000"
            " --log_dir test"
        )

        return [
            Job(
                name="train-job1",
                job_type="training",
                command=f"python {train_jobs_dir}/job1.py",
                gpu_idx=0,
                mps_percentage=50,
            ),
            Job(
                name="infer-recommend",
                job_type="inference",
                command=serve_recommend_cmd,
                gpu_idx=0,
                mps_percentage=50,
                envs={"PYTHONUNBUFFERED": "1"},
                readiness_log_marker="PEACE_EVENT: FIRST_BATCH_STARTED",
            ),
            Job(
                name="train-recommend",
                job_type="training",
                command=train_recommend_cmd,
                gpu_idx=0,
                mps_percentage=50,
                envs={
                    "PYTHONUNBUFFERED": "1",
                    "PEACE_CHECKPOINT_PATH": checkpoint_path,
                },
                workdir=train_workdir,
            ),
            Job(
                name="infer-job1",
                job_type="inference",
                command=f"python {serve_jobs_dir}/job1.py",
                gpu_idx=0,
                mps_percentage=50,
            ),
            Job(
                name="train-job2",
                job_type="training",
                command=f"python {train_jobs_dir}/job2.py",
                gpu_idx=0,
                mps_percentage=50,
            ),
            Job(
                name="infer-job2",
                job_type="inference",
                command=f"python {serve_jobs_dir}/job2.py",
                gpu_idx=0,
                mps_percentage=50,
            ),
            Job(
                name="train-job3",
                job_type="training",
                command=f"python {train_jobs_dir}/job3.py",
                gpu_idx=0,
                mps_percentage=50,
            ),
            Job(
                name="infer-job3",
                job_type="inference",
                command=f"python {serve_jobs_dir}/job3.py",
                gpu_idx=0,
                mps_percentage=50,
            ),
        ]

    def has_jobs(self) -> bool:
        return bool(self.job_queue)

    def peek_next_jobs(self, count: int = 1) -> List[Job]:
        return list(self.job_queue)[:count]

    def pop_next_job(self) -> Job:
        return self.job_queue.popleft()

    def refresh_node_state(self) -> PeaceNodeState:
        self.last_node_state = Monitor.get_peace_node_state(name_prefix=self.peace_prefix)
        return self.last_node_state

    def make_container_name(self, job: Job) -> str:
        if job.name.startswith(self.peace_prefix):
            return job.name
        return f"{self.peace_prefix}{job.name}"

    def start_job(self, job: Job) -> str:
        container_name = self.make_container_name(job)
        container_id = DockerLayer.start_container(
            self.image_name,
            container_name,
            job.command,
            job.gpu_idx,
            job.mps_percentage,
            self.volumes,
            envs=job.envs,
            workdir=job.workdir,
        )
        self.active_jobs_by_id[container_id] = job
        self.active_jobs_by_name[container_name] = job
        return container_id

    def make_redeploy_job(self, survivor_job: Job, next_job: Job) -> Job:
        generation = self.redeploy_generation.get(survivor_job.name, 0) + 1
        self.redeploy_generation[survivor_job.name] = generation

        envs = survivor_job.envs.copy()
        if survivor_job.job_type == "training":
            checkpoint_path = envs.get("PEACE_CHECKPOINT_PATH")
            if checkpoint_path:
                envs["PEACE_RESUME_PATH"] = checkpoint_path

        return Job(
            name=f"{survivor_job.name}-redeploy-{generation}",
            job_type=survivor_job.job_type,
            command=survivor_job.command,
            gpu_idx=survivor_job.gpu_idx,
            mps_percentage=max(1, 100 - next_job.mps_percentage),
            envs=envs,
            workdir=survivor_job.workdir,
            readiness_log_marker=survivor_job.readiness_log_marker,
        )

    def wait_until_container_absent_from_monitor(
        self,
        container_id: str,
        poll_interval: float = 0.5,
        stable_polls: int = 2,
        timeout: int = 30,
    ) -> PeaceNodeState:
        start_time = time.time()
        absent_polls = 0
        latest_state = self.refresh_node_state()

        while True:
            latest_state = self.refresh_node_state()
            running_ids = {job.container_id for job in latest_state.running_jobs}

            if container_id not in running_ids:
                absent_polls += 1
            else:
                absent_polls = 0

            if absent_polls >= stable_polls:
                logging.info(
                    "Scheduler: container %s is absent from PEACE monitor state.",
                    container_id,
                )
                return latest_state

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Timed out waiting for {container_id} to leave PEACE monitor state."
                )

            time.sleep(poll_interval)

    def wait_until_container_running(
        self,
        container_id: str,
        poll_interval: float = 0.2,
        timeout: int = 30,
    ) -> None:
        start_time = time.time()
        while True:
            if DockerLayer.is_container_running(container_id):
                logging.info("Scheduler: container %s is running.", container_id)
                return

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for {container_id} to start running.")

            time.sleep(poll_interval)

    def schedule_next_jobs(self, count: int) -> List[str]:
        scheduled_container_ids = []
        for _ in range(min(count, len(self.job_queue))):
            job = self.pop_next_job()
            container_id = self.start_job(job)
            scheduled_container_ids.append(container_id)
            logging.info(
                "Scheduler: scheduled %s (%s) as %s.",
                job.name,
                job.job_type,
                self.make_container_name(job),
            )

        return scheduled_container_ids

    def schedule_if_node_empty(self) -> List[str]:
        node_state = self.refresh_node_state()
        if node_state.running_count != 0:
            return []

        return self.schedule_next_jobs(2)

    def schedule_to_two_and_wait_for_exit(self) -> Optional[str]:
        """
        First-pass scheduling policy:
        - If 2 PEACE jobs are running, wait for either to exit.
        - If 1 PEACE job is running, schedule one queued job, then wait for either to exit.
        - If 0 PEACE jobs are running, schedule two queued jobs, then wait for either to exit.

        Returns the container id that exited, or None if there is no work to run/wait on.
        """
        node_state = self.refresh_node_state()

        if node_state.running_count >= 2:
            container_ids = [job.container_id for job in node_state.running_jobs]
            logging.info("Scheduler: %s PEACE jobs running. Waiting for one to exit.", node_state.running_count)
            return Monitor.wait_for_any_exit(container_ids)

        if node_state.running_count == 1:
            logging.info("Scheduler: one PEACE job running. Scheduling one more job.")
            scheduled_ids = self.schedule_next_jobs(1)
            if scheduled_ids:
                node_state = Monitor.wait_for_stable_peace_node_state(
                    name_prefix=self.peace_prefix,
                    expected_count=2,
                )

            container_ids = [job.container_id for job in node_state.running_jobs]
            if not container_ids:
                return None

            logging.info("Scheduler: waiting for one of %s to exit.", container_ids)
            return Monitor.wait_for_any_exit(container_ids)

        logging.info("Scheduler: no PEACE jobs running. Scheduling two jobs.")
        scheduled_ids = self.schedule_next_jobs(2)
        if not scheduled_ids:
            logging.info("Scheduler: no queued jobs available.")
            return None

        expected_count = len(scheduled_ids)
        node_state = Monitor.wait_for_stable_peace_node_state(
            name_prefix=self.peace_prefix,
            expected_count=expected_count,
        )
        container_ids = [job.container_id for job in node_state.running_jobs]
        if not container_ids:
            return None

        logging.info("Scheduler: waiting for one of %s to exit.", container_ids)
        return Monitor.wait_for_any_exit(container_ids)

    def handle_exit_and_trigger_workflow(self, exited_container_id: str) -> List[str]:
        """
        After any exit:
        - Remove the exited job from scheduler runtime state.
        - Find the surviving PEACE job.
        - Immediately schedule the next queued job.
        - Redeploy the survivor using the training or inference workflow.
        """
        exited_job = self.active_jobs_by_id.pop(exited_container_id, None)
        if exited_job:
            self.active_jobs_by_name.pop(self.make_container_name(exited_job), None)

        node_state = Monitor.wait_for_stable_peace_node_state(name_prefix=self.peace_prefix)
        if node_state.running_count == 0:
            logging.info("Scheduler: no survivor after exit. Scheduling two fresh jobs next.")
            return self.schedule_next_jobs(2)

        survivor = node_state.running_jobs[0]
        survivor_job = self.active_jobs_by_id.get(survivor.container_id)
        if survivor_job is None:
            survivor_job = self.active_jobs_by_name.get(survivor.container_name)

        if survivor_job is None:
            logging.error(
                "Scheduler: cannot identify survivor spec for %s. Workflow not triggered.",
                survivor.container_name,
            )
            return []

        if not self.job_queue:
            logging.info("Scheduler: no queued job to schedule alongside survivor.")
            return []

        next_job = self.pop_next_job()
        logging.info("Scheduler: immediately scheduling next queued job %s.", next_job.name)
        next_container_id = self.start_job(next_job)
        redeploy_job = self.make_redeploy_job(survivor_job, next_job)

        if survivor_job.job_type == "training":
            logging.info(
                "Scheduler: survivor %s is training. Sending checkpoint signal before redeploy.",
                survivor.container_name,
            )
            DockerLayer.send_signal(survivor.container_id, "SIGUSR1")
            Monitor.wait_for_any_exit([survivor.container_id])
            self.wait_until_container_absent_from_monitor(survivor.container_id)
            self.active_jobs_by_id.pop(survivor.container_id, None)
            self.active_jobs_by_name.pop(survivor.container_name, None)

            redeploy_container_id = self.start_job(redeploy_job)
            return [next_container_id, redeploy_container_id]

        logging.info(
            "Scheduler: survivor %s is inference. Starting redeploy before stopping old container.",
            survivor.container_name,
        )
        redeploy_container_id = self.start_job(redeploy_job)
        if redeploy_job.readiness_log_marker:
            Monitor.wait_for_log_message(redeploy_container_id, redeploy_job.readiness_log_marker)
        else:
            self.wait_until_container_running(redeploy_container_id)

        DockerLayer.kill_container(survivor.container_id)
        self.wait_until_container_absent_from_monitor(survivor.container_id)
        self.active_jobs_by_id.pop(survivor.container_id, None)
        self.active_jobs_by_name.pop(survivor.container_name, None)
        return [next_container_id, redeploy_container_id]


JobSpec = Job
