import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from docker_layer import DockerLayer
from monitor import Monitor, PeaceNodeState


@dataclass
class JobSpec:
    name: str
    job_type: str
    command: str
    gpu_idx: int
    mps_percentage: int
    envs: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    status: str = "queued"
    readiness_log_marker: Optional[str] = None
    replacement_target: Optional[str] = None
    replacement_strategy: Optional[str] = None
    checkpoint_signal: str = "SIGUSR1"


@dataclass
class ScheduledLaunch:
    job: JobSpec
    container_id: str
    container_name: str


@dataclass
class InFlightLaunch:
    job: JobSpec
    container_id: str
    container_name: str
    launched_at: float


class Scheduler:
    def __init__(
        self,
        image_name: str,
        volumes: Dict[str, Dict[str, str]],
        peace_prefix: str = "peace-",
        max_running_jobs: int = 2,
    ) -> None:
        self.image_name = image_name
        self.volumes = volumes
        self.peace_prefix = peace_prefix
        self.max_running_jobs = max_running_jobs
        self.pending_jobs: Deque[JobSpec] = deque()
        self.in_flight: List[InFlightLaunch] = []

    def enqueue(self, job: JobSpec) -> None:
        self.pending_jobs.append(job)

    def extend(self, jobs: List[JobSpec]) -> None:
        self.pending_jobs.extend(jobs)

    def has_pending_jobs(self) -> bool:
        return bool(self.pending_jobs)

    def has_in_flight_jobs(self) -> bool:
        return bool(self.in_flight)

    def has_work(self) -> bool:
        if self.pending_jobs or self.in_flight:
            return True
        return self.snapshot().running_count > 0

    def snapshot(self, stable: bool = True) -> PeaceNodeState:
        if stable:
            return Monitor.wait_for_stable_peace_node_state(name_prefix=self.peace_prefix)
        return Monitor.get_peace_node_state(name_prefix=self.peace_prefix)

    def make_container_name(self, job_name: str) -> str:
        # Ensure prefix is only added once
        if job_name.startswith(self.peace_prefix):
            return job_name
        return f"{self.peace_prefix}{job_name}"

    def _normalize_job_type(self, job_type: str) -> str:
        normalized = job_type.lower()
        if normalized in {"train", "training"}:
            return "training"
        if normalized in {"serve", "inference"}:
            return "inference"
        raise ValueError(f"Unsupported PEACE job_type: {job_type}")

    def _find_running_job(self, snapshot: PeaceNodeState, job_name: str):
        container_name = self.make_container_name(job_name)
        for running_job in snapshot.running_jobs:
            if running_job.container_name == container_name or running_job.job_name == job_name:
                return running_job
        return None

    def _prepare_replacement_if_needed(self, job: JobSpec, snapshot: PeaceNodeState) -> None:
        if not job.replacement_target:
            return

        target = self._find_running_job(snapshot, job.replacement_target)
        if target is None:
            logging.info(
                "Scheduler: replacement target %s is not running; launching %s normally.",
                job.replacement_target,
                job.name,
            )
            return

        job_type = self._normalize_job_type(job.job_type)
        strategy = job.replacement_strategy
        if strategy is None:
            strategy = "checkpoint-before-launch" if job_type == "training" else "ready-then-stop"

        if strategy == "checkpoint-before-launch":
            logging.info(
                "Scheduler: checkpointing %s before launching replacement %s.",
                target.container_name,
                job.name,
            )
            DockerLayer.send_signal(target.container_id, job.checkpoint_signal)
            Monitor.wait_for_any_exit([target.container_id])
            return

        if strategy == "ready-then-stop":
            logging.info(
                "Scheduler: %s will stay running until replacement %s is ready.",
                target.container_name,
                job.name,
            )
            return

        raise ValueError(f"Unsupported replacement_strategy: {strategy}")

    def _launch_jobs(self, jobs: List[JobSpec], snapshot: PeaceNodeState) -> List[ScheduledLaunch]:
        running_names = {job.container_name for job in snapshot.running_jobs}
        launched: List[ScheduledLaunch] = []

        for next_job in jobs:
            container_name = self.make_container_name(next_job.name)
            if container_name in running_names:
                logging.info("Scheduler: %s is already running; treating as settled.", container_name)
                next_job.status = "running"
                if self.pending_jobs and self.pending_jobs[0] is next_job:
                    self.pending_jobs.popleft()
                else:
                    try:
                        self.pending_jobs.remove(next_job)
                    except ValueError:
                        pass
                continue

            self._prepare_replacement_if_needed(next_job, snapshot)
            next_job.status = "inflight"
            container_id = DockerLayer.start_container(
                self.image_name,
                container_name,
                next_job.command,
                next_job.gpu_idx,
                next_job.mps_percentage,
                self.volumes,
                envs=next_job.envs,
                workdir=next_job.workdir,
            )
            launch = InFlightLaunch(
                job=next_job,
                container_id=container_id,
                container_name=container_name,
                launched_at=time.time(),
            )
            self.in_flight.append(launch)
            launched.append(
                ScheduledLaunch(
                    job=next_job,
                    container_id=container_id,
                    container_name=container_name,
                )
            )
            running_names.add(container_name)

        return launched

    def _is_launch_ready(self, launch: InFlightLaunch, snapshot: PeaceNodeState) -> bool:
        running_names = {job.container_name for job in snapshot.running_jobs}
        if launch.container_name not in running_names:
            return False

        job_type = self._normalize_job_type(launch.job.job_type)
        if job_type == "inference" and launch.job.readiness_log_marker:
            return DockerLayer.container_logs_contain(
                launch.container_id,
                launch.job.readiness_log_marker,
                tail=100,
            )

        return True

    def _commit_launch(self, launch: InFlightLaunch) -> None:
        launch.job.status = "running"

        if self.pending_jobs and self.pending_jobs[0] is launch.job:
            self.pending_jobs.popleft()
        else:
            try:
                self.pending_jobs.remove(launch.job)
            except ValueError:
                pass

        if launch.job.replacement_target:
            strategy = launch.job.replacement_strategy
            job_type = self._normalize_job_type(launch.job.job_type)
            if strategy is None:
                strategy = "checkpoint-before-launch" if job_type == "training" else "ready-then-stop"

            if strategy == "ready-then-stop":
                snapshot = self.snapshot()
                target = self._find_running_job(snapshot, launch.job.replacement_target)
                if target is not None:
                    logging.info(
                        "Scheduler: replacement %s is ready; stopping old container %s.",
                        launch.container_name,
                        target.container_name,
                    )
                    DockerLayer.stop_and_remove(target.container_id)

    def settle_in_flight(self) -> List[ScheduledLaunch]:
        if not self.in_flight:
            return []

        snapshot = self.snapshot()
        ready_launches = [
            launch for launch in self.in_flight if self._is_launch_ready(launch, snapshot)
        ]

        if len(ready_launches) != len(self.in_flight):
            waiting = [
                launch.container_name
                for launch in self.in_flight
                if launch not in ready_launches
            ]
            logging.info("Scheduler: Waiting for in-flight launch(es) to settle: %s", waiting)
            return []

        settled: List[ScheduledLaunch] = []
        for launch in ready_launches:
            self._commit_launch(launch)
            settled.append(
                ScheduledLaunch(
                    job=launch.job,
                    container_id=launch.container_id,
                    container_name=launch.container_name,
                )
            )

        self.in_flight.clear()
        logging.info(
            "Scheduler: In-flight launch(es) settled: %s",
            [launch.container_name for launch in settled],
        )
        return settled

    def reconcile(self, max_launches: int = 1) -> List[ScheduledLaunch]:
        """
        Apply the current node policy.

        Policy:
        - 2 or more PEACE containers running: do nothing.
        - 1 PEACE container running: launch the next queued job, if any.
        - 0 PEACE containers running: launch queued jobs up to the concurrency cap.

        Pending jobs are not removed from the queue until their launch settles.
        """
        if self.in_flight:
            self.settle_in_flight()
            return []

        snapshot = self.snapshot()
        available_slots = max(0, self.max_running_jobs - snapshot.running_count)
        launch_budget = min(max_launches, available_slots)

        if available_slots <= 0:
            logging.info(
                "Scheduler: %s PEACE job(s) already running; no scheduling action.",
                snapshot.running_count,
            )
            return []

        jobs_to_launch = list(self.pending_jobs)[:launch_budget]
        return self._launch_jobs(jobs_to_launch, snapshot)

    def step(self) -> List[ScheduledLaunch]:
        if self.in_flight:
            return self.settle_in_flight()

        snapshot = self.snapshot()
        if snapshot.running_count == 0:
            return self.reconcile(max_launches=self.max_running_jobs)

        if snapshot.running_count == 1:
            return self.reconcile(max_launches=1)

        logging.info(
            "Scheduler: %s PEACE job(s) running; no scheduling action.",
            snapshot.running_count,
        )
        return []
