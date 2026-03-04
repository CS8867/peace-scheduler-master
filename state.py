import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

# 1. Define Statuses (So we don't use magic strings)
class ContainerStatus(Enum):
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    STOPPED = "STOPPED" # Waiting to be cleaned up
    KILLED = "KILLED"

# 2. Define a clean Data Structure (No more confusing tuples)
@dataclass
class ContainerInfo:
    container_id: str
    job_tag: str          # e.g., "job1", "job2_old" - easier to find!
    server_idx: int
    gpu_idx: int
    mps_percentage: int   # NEW: Track how much GPU it uses
    status: ContainerStatus
    checkpoint_path: Optional[str] = None # NEW: Store where the file is

class State:
    # Map: container_id -> ContainerInfo
    containers: Dict[str, ContainerInfo]

    def __init__(self) -> None:
        logging.info("Initializing Enhanced State")
        self.containers = {}

    # NEW: Find a container by its easy name ("job1") instead of random ID
    def get_container_by_tag(self, tag: str) -> Optional[ContainerInfo]:
        for info in self.containers.values():
            if info.job_tag == tag:
                return info
        return None

    # UPGRADED: Now accepts mps_percentage and job_tag
    def assign(self, container_id: str, job_tag: str, server_idx: int, gpu_idx: int, mps_percentage: int):
        info = ContainerInfo(
            container_id=container_id,
            job_tag=job_tag,
            server_idx=server_idx,
            gpu_idx=gpu_idx,
            mps_percentage=mps_percentage,
            status=ContainerStatus.RUNNING
        )
        self.containers[container_id] = info
        logging.info(f"Assigned {job_tag} ({container_id}) to GPU {gpu_idx} with {mps_percentage}%")

    # NEW: Update status (e.g., set to CHECKPOINTING)
    def update_status(self, container_id: str, status: ContainerStatus, ckpt_path: str = None):
        if container_id in self.containers:
            self.containers[container_id].status = status
            if ckpt_path:
                self.containers[container_id].checkpoint_path = ckpt_path
            logging.info(f"Container {container_id} status changed to {status}")

    def remove(self, container_id: str):
        if container_id in self.containers:
            tag = self.containers[container_id].job_tag
            del self.containers[container_id]
            logging.info(f"Removed {tag} ({container_id}) from state")

# Singleton Pattern (Kept from original)
_state_instance: State = State()

def get_state() -> State:
    return _state_instance