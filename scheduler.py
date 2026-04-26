import logging
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


@dataclass
class ScheduledLaunch:
    job: JobSpec
    container_id: str
    container_name: str


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

    def enqueue(self, job: JobSpec) -> None:
        self.pending_jobs.append(job)

    def extend(self, jobs: List[JobSpec]) -> None:
        self.pending_jobs.extend(jobs)

    def has_pending_jobs(self) -> bool:
        return bool(self.pending_jobs)

    def snapshot(self) -> PeaceNodeState:
        return Monitor.get_peace_node_state(name_prefix=self.peace_prefix)

    def make_container_name(self, job_name: str) -> str:
        # Ensure prefix is only added once
        if job_name.startswith(self.peace_prefix):
            return job_name
        return f"{self.peace_prefix}{job_name}"

    def reconcile(self) -> List[ScheduledLaunch]:
        """
        Fill open PEACE slots up to the configured concurrency cap.
        """
        snapshot = self.snapshot()
        available_slots = max(0, self.max_running_jobs - snapshot.running_count)
        running_names = {job.container_name for job in snapshot.running_jobs}
        launched: List[ScheduledLaunch] = []

        while available_slots > 0 and self.pending_jobs:
            next_job = self.pending_jobs.popleft()
            container_name = self.make_container_name(next_job.name)
            if container_name in running_names:
                logging.info("Scheduler: %s is already running; skipping duplicate launch.", container_name)
                continue

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
            launched.append(
                ScheduledLaunch(
                    job=next_job,
                    container_id=container_id,
                    container_name=container_name,
                )
            )
            running_names.add(container_name)
            available_slots -= 1

        return launched