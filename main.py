import time
import logging
import sys
import os
from typing import List
from state import get_state, ContainerStatus
from docker_layer import DockerLayer
from monitor import Monitor
from router import Router
from scheduler import JobSpec, Scheduler
import argparse

# Import your workflows
# from src.pipelines.training_pipeline import run_training
# from src.serving.api_server import start_api_server

# --- CONFIGURATION ---
DEBUG_MODE = True  # Set to True to KEEP containers after run (for debugging)
CHECKPOINT_HOST_DIR = "/home/node4/mlProfiler/checkpoints" 
CHECKPOINT_MOUNT_DIR = "/app/checkpoints"
IMAGE_NAME = "nba556677/ml_tasks:latest"
# WORK_DIR = "/home/node4/peace-scheduler-master/jobs"
TRAIN_JOBS_DIR = "/home/node4/peace-repo/peace-scheduler-master/train-jobs"
SERVE_JOBS_DIR = "/home/node4/peace-repo/peace-scheduler-master/jobs"
TRAIN_CONTAINER_JOBS_DIR = "/app/train-jobs"
SERVE_CONTAINER_JOBS_DIR = "/app/serve-jobs"
# FRAMEWORK_DIR = "/home/node4/peace-scheduler-master"

# --- SERVE WORKLOAD CONFIGURATION ---
# Volumes mirroring: docker run ... -v /tmp/nvidia-mps:/tmp/nvidia-mps
#   -v ~/.cache/huggingface:/root/.cache/huggingface
#   -v ~/.cache/torch:/root/.cache/torch
#   -v ~/mlProfiler:/root/mlprofiler
#   -v /opt/nvidia/nsight-systems/2023.3.3:/nsys
HOME_DIR = os.path.expanduser("~")
# SERVE_VOLUMES = {
#     '/tmp/nvidia-mps':                            {'bind': '/tmp/nvidia-mps', 'mode': 'rw'},
#     os.path.join(HOME_DIR, '.cache/huggingface'): {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
#     os.path.join(HOME_DIR, '.cache/torch'):       {'bind': '/root/.cache/torch', 'mode': 'rw'},
#     os.path.join(HOME_DIR, 'mlProfiler'):         {'bind': '/root/mlprofiler', 'mode': 'rw'},
#     '/opt/nvidia/nsight-systems/2023.3.3':        {'bind': '/nsys', 'mode': 'ro'},
# }

# The inference command to run inside the container
SERVE_INFERENCE_CMD = (
    "python recommend-inference.py"
    " --batch_size 2"
    " --model_name bert-base-cased"
    " --profile_nstep 10000"
    " --log_dir test"
)
TRAIN_RECOMMEND_CMD = (
    f"python {TRAIN_CONTAINER_JOBS_DIR}/recommend-train.py"
    " --batch_size 2"
    " --model_name bert-large-cased"
    " --profile_nstep 1000"
    " --log_dir test"
)
TRAIN_JOB2_CHECKPOINT = f"{CHECKPOINT_MOUNT_DIR}/job2_ckpt.pt"
TRAIN_WORKDIR = "/root/mlprofiler/workloads/train"
SERVE_WORKDIR = "/root/mlprofiler/workloads/inference"
TRAIN_RECOMMEND_CHECKPOINT = f"{CHECKPOINT_MOUNT_DIR}/recommend_train_ckpt.pt"
FIRST_BATCH_LOG_MARKER = "PEACE_EVENT: FIRST_BATCH_STARTED"
PEACE_CONTAINER_PREFIX = "peace-"

# --- LOGGING SETUP (CRITICAL FIX) ---
# We force logging to stream to Standard Out so you see it in the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MAIN - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

def debug_logs(container_id, name):
    """Helper to dump container logs to our main terminal"""
    if DEBUG_MODE:
        try:
            import docker
            client = docker.from_env()
            c = client.containers.get(container_id)
            logger.info(f"--- LOGS FOR {name} ({container_id}) ---")
            logger.info(c.logs().decode('utf-8'))
            logger.info("------------------------------------------")
        except:
            pass


def build_dynamic_training_jobs() -> List[JobSpec]:
    """
    Logical training queue. The scheduler redeploys the surviving training job
    internally when a slot opens; the queue does not contain "*_new" jobs.
    """
    return [
        JobSpec(
            name="train-job1",
            job_type="training",
            command=f"python {TRAIN_CONTAINER_JOBS_DIR}/job1.py",
            gpu_idx=0,
            mps_percentage=50,
        ),
        JobSpec(
            name="train-recommend",
            job_type="training",
            command=TRAIN_RECOMMEND_CMD,
            gpu_idx=0,
            mps_percentage=50,
            envs={
                "PYTHONUNBUFFERED": "1",
                "PEACE_CHECKPOINT_PATH": TRAIN_RECOMMEND_CHECKPOINT,
            },
            workdir=TRAIN_WORKDIR,
        ),
        JobSpec(
            name="train-job3",
            job_type="training",
            command=f"python {TRAIN_CONTAINER_JOBS_DIR}/job3.py",
            gpu_idx=0,
            mps_percentage=60,
        ),
    ]


def build_dynamic_inference_jobs() -> List[JobSpec]:
    """
    Logical inference queue. The scheduler redeploys the surviving inference job
    internally when a slot opens; the queue does not contain "*_new" jobs.
    """
    serve_job_cmd = (
        f"bash -c 'cd {SERVE_WORKDIR} && python {SERVE_CONTAINER_JOBS_DIR}/recommend-inference.py"
        " --batch_size 2"
        " --model_name bert-base-cased"
        " --profile_nstep 10000"
        " --log_dir test'"
    )

    return [
        JobSpec(
            name="train-job1",
            job_type="training",
            command=f"python {TRAIN_CONTAINER_JOBS_DIR}/job1.py",
            gpu_idx=0,
            mps_percentage=50,
        ),
        JobSpec(
            name="serve-recommend",
            job_type="inference",
            command=serve_job_cmd,
            gpu_idx=0,
            mps_percentage=50,
            envs={"PYTHONUNBUFFERED": "1"},
            readiness_log_marker=FIRST_BATCH_LOG_MARKER,
        ),
        JobSpec(
            name="train-job3",
            job_type="training",
            command=f"python {TRAIN_CONTAINER_JOBS_DIR}/job3.py",
            gpu_idx=0,
            mps_percentage=60,
        ),
    ]


def build_dynamic_jobs(workflow: str) -> List[JobSpec]:
    if workflow == "training":
        return build_dynamic_training_jobs()
    if workflow == "inference":
        return build_dynamic_inference_jobs()
    raise ValueError(f"Unsupported dynamic workflow: {workflow}")



def main():
        
    parser = argparse.ArgumentParser(description="ML System Entry Point")
    
    # Create a 'mode' argument to switch between workflows. --mode train will activate the train workflow, while --mode serve will activate the real-time inference workflow.
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'dynamic-train', 'serve', 'serve-gpu-check', 'serve-log-check', 'inference', 'general', 'monitor-check'], 
        required=True,
        help="Select 'train' for the current role-based training workflow, 'dynamic-train' for the queue-driven scheduler prototype, 'serve' for real-time inference API, 'serve-gpu-check' for serve with GPU activity probing, 'serve-log-check' for serve with custom first-batch log readiness, or 'inference' for standalone inference workload"
    )

    # Optional: specific configs for each mode
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config file")
    parser.add_argument(
        '--dynamic-workflow',
        type=str,
        choices=['training', 'inference'],
        default='training',
        help="Select which replacement workflow the dynamic scheduler should exercise."
    )

    args = parser.parse_args()

    state = get_state()

    # Volumes Dictionary to specify what host directories we mount into our containers. The 'bind' keyword takes the directory as an argument. The mode specifies read/write modes.
    volumes = {
        CHECKPOINT_HOST_DIR: {'bind': CHECKPOINT_MOUNT_DIR, 'mode': 'rw'},
        '/tmp/nvidia-mps': {'bind': '/tmp/nvidia-mps', 'mode': 'rw'},
        TRAIN_JOBS_DIR: {'bind': TRAIN_CONTAINER_JOBS_DIR, 'mode': 'ro'},
        SERVE_JOBS_DIR: {'bind': SERVE_CONTAINER_JOBS_DIR, 'mode': 'ro'},
        '/tmp/nvidia-mps':                            {'bind': '/tmp/nvidia-mps', 'mode': 'rw'},
        os.path.join(HOME_DIR, '.cache/huggingface'): {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
        os.path.join(HOME_DIR, '.cache/torch'):       {'bind': '/root/.cache/torch', 'mode': 'rw'},
        os.path.join(HOME_DIR, 'mlProfiler'):         {'bind': '/root/mlprofiler', 'mode': 'rw'},
        '/opt/nvidia/nsight-systems/2023.3.3':        {'bind': '/nsys', 'mode': 'ro'}
        # '/home/node4/peace-scheduler-master/framework.py': {'bind': CONTAINER_JOBS_DIR, 'mode': 'ro'}  # If your framework code is in a separate directory
    }
    


    # --- Orchestration Logic ---
    if args.mode == 'train':
        # 1. DEFINE COMMANDS
        cmd_job1 = f"python {TRAIN_CONTAINER_JOBS_DIR}/job1.py"
        cmd_job2_old = (
            f"python {TRAIN_CONTAINER_JOBS_DIR}/recommend-train.py"
            " --batch_size 2"
            " --model_name bert-large-cased"
            " --profile_nstep 1000"
            " --log_dir test"
        )
        cmd_job2_new = cmd_job2_old
        train_job2_old_envs = {
            "PYTHONUNBUFFERED": "1",
            "PEACE_CHECKPOINT_PATH": TRAIN_RECOMMEND_CHECKPOINT,
        }
        train_job2_new_envs = {
            "PYTHONUNBUFFERED": "1",
            "PEACE_CHECKPOINT_PATH": TRAIN_RECOMMEND_CHECKPOINT,
            "PEACE_RESUME_PATH": TRAIN_RECOMMEND_CHECKPOINT,
        }
        cmd_job3 = f"python {TRAIN_CONTAINER_JOBS_DIR}/job3.py"
        print("Starting Training Workflow...")

        # In a real scenario, you might pass the config path here
        # run_training(config_path=args.config)
        logger.info(">>> Launching Initial Training Jobs (Job1 + Job2 Old)...")

        # --- WORKFLOW TIMER: start when job2_old launches ---
        # workflow_start_time = time.time()

        # 1. Start Initial Jobs
        job1_id = DockerLayer.start_container(IMAGE_NAME, "peace-job1", cmd_job1, 0, 50, volumes)
        job2_old_id = DockerLayer.start_container(
            IMAGE_NAME, "peace-job2_old", cmd_job2_old, 0, 50, volumes, envs=train_job2_old_envs, workdir=TRAIN_WORKDIR
        )

        controller_waiting_for_job1_exit_start = time.time()
        # logger.info(f"[TIMER] Time printed after job1_id and job2_old_id have started: {time_after_start_initial_containers - workflow_start_time:.4f} seconds")

        # 2. Wait for Job 1 to Exit
        logger.info("Waiting for Job 1 to finish...")
        Monitor.wait_for_any_exit([job1_id])
        logger.info("Job 1 finished.")
        debug_logs(job1_id, "job1")

        controller_waiting_for_job1_exit_end = time.time()
        logger.info(f"[TIMER] Time printed after job1 finishes: {controller_waiting_for_job1_exit_end - controller_waiting_for_job1_exit_start:.4f} seconds")

        if not DockerLayer.is_container_running(job2_old_id):
            logger.info("Job 2 Old finished before Job 1. Skipping replacement spawn.")
            debug_logs(job2_old_id, "job2_old")
            logger.info("Training Workflow Complete.")
            return

        # 3. Ask Job 2 Old to checkpoint so the replacement workload can resume.
        logger.info(f"Signaling Job 2 Old ({job2_old_id}) to checkpoint and exit...")
        
        # Send a signal so the current training workload saves its state before Phase 2 starts.
        DockerLayer.send_signal(job2_old_id, "SIGUSR1") 
        
        time_after_sending_signal = time.time()
        logger.info(f"[TIMER] Time printed after sending signal to job2_old: {time_after_sending_signal - controller_waiting_for_job1_exit_end:.4f} seconds")

        # Wait for it to checkpoint and terminate before starting the replacement container.
        Monitor.wait_for_any_exit([job2_old_id])
        logger.info("Job 2 Old has successfully checkpointed and exited.")
        debug_logs(job2_old_id, "job2_old")

        time_after_job2_old_exits = time.time()
        logger.info(f"[TIMER] Time printed after job2_old finishes: {time_after_job2_old_exits - time_after_sending_signal:.4f} seconds")

        # 4. Start Next Phase
        logger.info(">>> Launching Phase 2 (Job 2 New + Job 3)...")
        job2_new_id = DockerLayer.start_container(
            IMAGE_NAME, "peace-job2_new", cmd_job2_new, 0, 40, volumes, envs=train_job2_new_envs, workdir=TRAIN_WORKDIR
        )
        job3_id = DockerLayer.start_container(IMAGE_NAME, "peace-job3", cmd_job3, 0, 60, volumes)

        time_after_starting_new_containers = time.time()
        logger.info(f"[TIMER] Time printed after starting job2_new and job3: {time_after_starting_new_containers - time_after_job2_old_exits:.4f} seconds")
        
        # Optional: Monitor them to completion
        Monitor.wait_for_any_exit([job2_new_id])

        # --- WORKFLOW TIMER: end when job2_new finishes ---
        # workflow_end_time = time.time()
        # logger.info("[TIMER] Time printed after job2_new finishes: {:.4f} seconds".format(workflow_end_time - time_after_starting_new_containers))
        # workflow_duration = workflow_end_time - workflow_start_time
        # logger.info(f"[TIMER] training_workflow_time (job2_old start -> job2_new end): {workflow_duration:.4f} seconds")

        Monitor.wait_for_any_exit([job3_id])
        
        # Dump logs to confirm the resumed container and the colocated job both completed.
        debug_logs(job2_new_id, "job2_new")
        debug_logs(job3_id, "job3")
        
        # Cleanup
        DockerLayer.stop_and_remove(job2_new_id)
        DockerLayer.stop_and_remove(job3_id)
        logger.info("Training Workflow Complete.")

    elif args.mode == 'dynamic-train':
        logger.info(">>> Starting Dynamic Queue Scheduler...")
        scheduler = Scheduler(
            image_name=IMAGE_NAME,
            volumes=volumes,
            peace_prefix=PEACE_CONTAINER_PREFIX,
        )

        while True:
            exited_container_id = scheduler.schedule_to_two_and_wait_for_exit()
            if exited_container_id is None:
                logger.info("Dynamic Queue Scheduler has no more work to schedule or monitor.")
                break

            logger.info("Scheduler observed exit for container %s.", exited_container_id)
            launched_ids = scheduler.handle_exit_and_trigger_workflow(exited_container_id)
            if launched_ids:
                logger.info("Scheduler launched workflow containers: %s", launched_ids)

        logger.info("Dynamic Queue Scheduler Complete.")

    elif args.mode == 'serve-gpu-check':
        # ----------------------------------------------------------------
        # Serve workflow with GPU activity probing + router switch timing.
        #   Same as 'serve', but instead of checking container.status == running,
        #   we wait until the workload is actually using the GPU (nvidia-smi)
        #   before switching the router and killing job2_old.
        #   Phase 1: job1 (toy, 50%) + job2_old (inference, 50%)
        #   Phase 2: job2_new (inference, 40%) + job3 (toy, 60%)
        # ----------------------------------------------------------------


        # Build the commands for serve-gpu-check mode (all scripts from jobs/ folder)
        serve_job2_cmd = f"bash -c 'cd /root/mlprofiler/workloads/inference && python {SERVE_CONTAINER_JOBS_DIR}/recommend-inference.py --batch_size 2 --model_name bert-base-cased --profile_nstep 10000 --log_dir test'"
        serve_job1_cmd = f"python {SERVE_CONTAINER_JOBS_DIR}/job1.py"
        serve_job3_cmd = f"python {SERVE_CONTAINER_JOBS_DIR}/job3.py"
        serve_envs = {"PYTHONUNBUFFERED": "1"}

        # Create the router
        router = Router()

        # 1. START INITIAL STATE: Job1 (toy) + Job2_old (inference) at 50/50 MPS
        logger.info(">>> [GPU-Check] Launching Initial Jobs (Job1 + Job2 Old)...")
        job1_id = DockerLayer.start_container(IMAGE_NAME, "peace-job1", serve_job1_cmd, 0, 50, volumes, envs=serve_envs)
        job2_old_id = DockerLayer.start_container(
            IMAGE_NAME, "peace-job2_old", serve_job2_cmd, 0, 50, volumes, envs=serve_envs
        )

        # Point router to job2_old
        router.set_backend(job2_old_id)
        logger.info("Router -> Job2 Old")

        # 2. Wait for Job1 to complete
        logger.info("Waiting for Job 1 to finish...")
        Monitor.wait_for_any_exit([job1_id])
        logger.info("Job 1 finished.")

        # 3. Spawn Job2_new and Job3 with revised MPS partitions
        logger.info(">>> [GPU-Check] Launching Phase 2 (Job2 New + Job 3)...")
        job2_new_id = DockerLayer.start_container(
            IMAGE_NAME, "peace-job2_new", serve_job2_cmd, 0, 40, volumes, envs=serve_envs
        )
        job3_id = DockerLayer.start_container(IMAGE_NAME, "peace-job3", serve_job3_cmd, 0, 60, volumes, envs=serve_envs)

        # 4. Wait for Job2_new to actually use the GPU before switching
        start_time = time.time()
        result = Monitor.wait_for_gpu_run(job2_new_id)
        total_time = time.time() - start_time

        if result is None:
            logger.error("Job2 New never started using the GPU. Aborting swap.")
            debug_logs(job2_new_id, "job2_new")
        else:
            logger.info(f"Job2 New confirmed on GPU after {total_time:.2f}s.")

            # 5. GPU confirmed — switch router to job2_new and measure the switch time
            switch_start = time.time()
            router.set_backend(job2_new_id)
            switch_duration = time.time() - switch_start
            logger.info(f">>> ROUTER SWITCH DOWNTIME: {switch_duration:.6f} seconds")

            # 6. Kill Job2_old now that router points to Job2_new
            debug_logs(job2_old_id, "job2_old")
            DockerLayer.stop_and_remove(job2_old_id)
            logger.info("Job2 Old stopped and removed.")

        # 7. Dump logs from new containers
        debug_logs(job2_new_id, "job2_new")
        debug_logs(job3_id, "job3")

        # 8. Verify Job2_old is gone
        if not DockerLayer.is_container_running(job2_old_id):
            logger.info("Serve-GPU-Check Workflow Complete. Job2_old successfully replaced.")
        else:
            logger.error("Job2_old is still running! Something went wrong.")

    elif args.mode == 'serve-log-check':
        # ----------------------------------------------------------------
        # Serve workflow with log-based readiness probing.
        #   Phase 1: job1 (toy, 50%) + job2_old (inference, 50%)
        #   Phase 2: job2_new (inference, 40%) + job3 (toy, 60%)
        #   Switch only after job2_new emits a custom first-batch marker.
        # ----------------------------------------------------------------
        serve_job2_cmd = f"bash -c 'cd /root/mlprofiler/workloads/inference && python {SERVE_CONTAINER_JOBS_DIR}/recommend-inference.py --batch_size 2 --model_name bert-base-cased --profile_nstep 10000 --log_dir test'"
        serve_job1_cmd = f"python {SERVE_CONTAINER_JOBS_DIR}/job1.py"
        serve_job3_cmd = f"python {SERVE_CONTAINER_JOBS_DIR}/job3.py"
        serve_envs = {"PYTHONUNBUFFERED": "1"}

        router = Router()

        logger.info(">>> [Log-Check] Launching Initial Jobs (Job1 + Job2 Old)...")
        job1_id = DockerLayer.start_container(IMAGE_NAME, "job1", serve_job1_cmd, 0, 50, volumes, envs=serve_envs)
        job2_old_id = DockerLayer.start_container(
            IMAGE_NAME, "job2_old", serve_job2_cmd, 0, 50, volumes, envs=serve_envs
        )

        router.set_backend(job2_old_id)
        logger.info("Router -> Job2 Old")

        logger.info("Waiting for Job 1 to finish...")
        Monitor.wait_for_any_exit([job1_id])
        logger.info("Job 1 finished.")

        logger.info(">>> [Log-Check] Launching Phase 2 (Job2 New + Job 3)...")
        job2_new_id = DockerLayer.start_container(
            IMAGE_NAME, "job2_new", serve_job2_cmd, 0, 40, volumes, envs=serve_envs
        )
        job3_id = DockerLayer.start_container(IMAGE_NAME, "job3", serve_job3_cmd, 0, 60, volumes, envs=serve_envs)

        start_time = time.time()
        result = Monitor.wait_for_log_message(job2_new_id, FIRST_BATCH_LOG_MARKER)
        total_time = time.time() - start_time

        if result is None:
            logger.error("Job2 New never emitted the first-batch marker. Aborting swap.")
            debug_logs(job2_new_id, "job2_new")
        else:
            logger.info(f"Job2 New emitted the first-batch marker after {total_time:.2f}s.")

            switch_start = time.time()
            router.set_backend(job2_new_id)
            switch_duration = time.time() - switch_start
            logger.info(f">>> ROUTER SWITCH DOWNTIME: {switch_duration:.6f} seconds")

            debug_logs(job2_old_id, "job2_old")
            DockerLayer.stop_and_remove(job2_old_id)
            logger.info("Job2 Old stopped and removed.")

        debug_logs(job2_new_id, "job2_new")
        debug_logs(job3_id, "job3")

        if not DockerLayer.is_container_running(job2_old_id):
            logger.info("Serve-Log-Check Workflow Complete. Job2_old successfully replaced.")
        else:
            logger.error("Job2_old is still running! Something went wrong.")

    elif args.mode == 'inference':
        # ----------------------------------------------------------------
        # Standalone inference: two-step docker workflow
        #   Step 1: Start an interactive container (docker run -it ... bash)
        #   Step 2: Exec the inference command inside it
        # ----------------------------------------------------------------
        logger.info(">>> Starting Standalone Inference Workload...")

        # Step 1 – Start the container in interactive mode with 'bash'
        #   This mirrors: docker run --rm -it --name w2 \
        #     --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 \
        #     --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=all \
        #     -v ... nba556677/ml_tasks:latest bash
        container_id = DockerLayer.start_container(
            image=IMAGE_NAME,
            name="peace-w2",
            command="bash",          # keeps the container alive
            gpu_idx=0,
            mps_percentage=50,
            volumes=volumes,
            interactive=True,        # -it flags
        )
        logger.info(f"Container 'w2' started: {container_id}")

        # Step 2 – Execute the workload command inside the running container
        #   This mirrors: docker exec w2 bash -c "cd /root/mlprofiler/workloads/inference && python ..."
        exit_code, output = DockerLayer.exec_command(
            container_id=container_id,
            command=SERVE_INFERENCE_CMD,
            workdir=SERVE_WORKDIR,
        )

        if exit_code == 0:
            logger.info("Inference workload completed successfully.")
        else:
            logger.error(f"Inference workload failed with exit code {exit_code}.")

        debug_logs(container_id, "w2")

        # Cleanup
        DockerLayer.stop_and_remove(container_id)
        logger.info("Inference Workflow Complete.")

    elif args.mode == 'general':
        logger.info(">>> Starting General Monitor Test Mode...")
        # Launch two PEACE jobs (simple dummy jobs)
        job1_id = DockerLayer.start_container(IMAGE_NAME, "peace-jobA", f"python {TRAIN_CONTAINER_JOBS_DIR}/job1.py", 0, 50, volumes)
        job2_id = DockerLayer.start_container(IMAGE_NAME, "peace-jobB", f"python {TRAIN_CONTAINER_JOBS_DIR}/job3.py", 0, 50, volumes)

        # Wait until Docker reports a stable PEACE view before reading state.
        node_state = Monitor.wait_for_stable_peace_node_state(expected_count=2)
        running_count = Monitor.get_peace_running_job_count()
        logger.info(f"[GENERAL MODE] PEACE node state: {node_state}")
        logger.info(f"[GENERAL MODE] PEACE running job count: {running_count}")

        # Cleanup
        DockerLayer.stop_and_remove(job1_id)
        DockerLayer.stop_and_remove(job2_id)
        logger.info("General Monitor Test Complete.")

    elif args.mode == 'monitor-check':
        logger.info(">>> Starting Monitor Check Mode...")
        job1_id = DockerLayer.start_container(
            IMAGE_NAME,
            "peace-monitor-train-job1",
            f"python {TRAIN_CONTAINER_JOBS_DIR}/job1.py",
            0,
            50,
            volumes,
        )
        job2_id = DockerLayer.start_container(
            IMAGE_NAME,
            "peace-monitor-infer-job3",
            f"python {SERVE_CONTAINER_JOBS_DIR}/job3.py",
            0,
            50,
            volumes,
        )

        node_state = Monitor.wait_for_stable_peace_node_state(expected_count=2)
        logger.info("[MONITOR CHECK] running_count=%s", node_state.running_count)
        for job in node_state.running_jobs:
            logger.info(
                "[MONITOR CHECK] container_id=%s name=%s job_name=%s status=%s gpu_idx=%s mps=%s",
                job.container_id,
                job.container_name,
                job.job_name,
                job.status,
                job.gpu_idx,
                job.mps_percentage,
            )

        DockerLayer.stop_and_remove(job1_id)
        DockerLayer.stop_and_remove(job2_id)
        logger.info("Monitor Check Complete.")

if __name__ == "__main__":
    main()
