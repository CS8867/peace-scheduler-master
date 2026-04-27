from collections import deque
from dataclasses import dataclass, field
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
        return DockerLayer.start_container(
            self.image_name,
            self.make_container_name(job),
            job.command,
            job.gpu_idx,
            job.mps_percentage,
            self.volumes,
            envs=job.envs,
            workdir=job.workdir,
        )

    def schedule_if_node_empty(self) -> List[str]:
        node_state = self.refresh_node_state()
        if node_state.running_count != 0:
            return []

        scheduled_container_ids = []
        for _ in range(min(2, len(self.job_queue))):
            job = self.pop_next_job()
            container_id = self.start_job(job)
            scheduled_container_ids.append(container_id)

        return scheduled_container_ids


JobSpec = Job
