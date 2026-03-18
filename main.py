import time
import logging
import sys
import os
from state import get_state, ContainerStatus
from docker_layer import DockerLayer
from monitor import Monitor
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
HOST_JOBS_DIR = "/home/node4/peace-scheduler-master/train-jobs"
CONTAINER_JOBS_DIR = "/app/jobs"
# FRAMEWORK_DIR = "/home/node4/peace-scheduler-master"

# --- SERVE WORKLOAD CONFIGURATION ---
# Volumes mirroring: docker run ... -v /tmp/nvidia-mps:/tmp/nvidia-mps
#   -v ~/.cache/huggingface:/root/.cache/huggingface
#   -v ~/.cache/torch:/root/.cache/torch
#   -v ~/mlProfiler:/root/mlprofiler
#   -v /opt/nvidia/nsight-systems/2023.3.3:/nsys
HOME_DIR = os.path.expanduser("~")
SERVE_VOLUMES = {
    '/tmp/nvidia-mps':                            {'bind': '/tmp/nvidia-mps', 'mode': 'rw'},
    os.path.join(HOME_DIR, '.cache/huggingface'): {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
    os.path.join(HOME_DIR, '.cache/torch'):       {'bind': '/root/.cache/torch', 'mode': 'rw'},
    os.path.join(HOME_DIR, 'mlProfiler'):         {'bind': '/root/mlprofiler', 'mode': 'rw'},
    '/opt/nvidia/nsight-systems/2023.3.3':        {'bind': '/nsys', 'mode': 'ro'},
}

# The inference command to run inside the container
SERVE_INFERENCE_CMD = (
    "python recommend-inference.py"
    " --batch_size 2"
    " --model_name bert-base-cased"
    " --profile_nstep 100"
    " --log_dir test"
)
SERVE_WORKDIR = "/root/mlprofiler/workloads/inference"

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



def main():
    parser = argparse.ArgumentParser(description="ML System Entry Point")
    
    # Create a 'mode' argument to switch between workflows. --mode train will activate the train workflow, while --mode serve will activate the real-time inference workflow.
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'serve', 'serve-gpu-check', 'inference'], 
        required=True,
        help="Select 'train' for model training, 'serve' for real-time inference API, 'serve-gpu-check' for serve with GPU activity probing, or 'inference' for standalone inference workload"
    )

    # Optional: specific configs for each mode
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config file")

    args = parser.parse_args()

    state = get_state()

    # Volumes Dictionary to specify what host directories we mount into our containers. The 'bind' keyword takes the directory as an argument. The mode specifies read/write modes.
    volumes = {
        CHECKPOINT_HOST_DIR: {'bind': CHECKPOINT_MOUNT_DIR, 'mode': 'rw'},
        '/tmp/nvidia-mps': {'bind': '/tmp/nvidia-mps', 'mode': 'rw'},
        HOST_JOBS_DIR: {'bind': CONTAINER_JOBS_DIR, 'mode': 'ro'},
        # '/home/node4/peace-scheduler-master/framework.py': {'bind': CONTAINER_JOBS_DIR, 'mode': 'ro'}  # If your framework code is in a separate directory
    }
    
    # 1. DEFINE COMMANDS
    cmd_job1 = f"python {CONTAINER_JOBS_DIR}/job1.py"
    cmd_job2_old = f"python {CONTAINER_JOBS_DIR}/job2.py --save_path {CHECKPOINT_MOUNT_DIR}/job2_ckpt.pt"
    cmd_job2_new = f"python {CONTAINER_JOBS_DIR}/job2.py --resume_from {CHECKPOINT_MOUNT_DIR}/job2_ckpt.pt --max_epochs 30"
    cmd_job3 = f"python {CONTAINER_JOBS_DIR}/job3.py"

    # --- Orchestration Logic ---
    if args.mode == 'train':
        print("🚀 Starting Training Workflow...")
        # In a real scenario, you might pass the config path here
        # run_training(config_path=args.config)
        logger.info(">>> Launching Initial Training Jobs (Job1 + Job2 Old)...")

        # 1. Start Initial Jobs
        job1_id = DockerLayer.start_container(IMAGE_NAME, "job1", cmd_job1, 0, 50, volumes)
        job2_old_id = DockerLayer.start_container(IMAGE_NAME, "job2_old", cmd_job2_old, 0, 50, volumes)

        # 2. Wait for Job 1 to Exit
        logger.info("Waiting for Job 1 to finish...")
        Monitor.wait_for_any_exit([job1_id])
        logger.info("Job 1 finished.")

        # 3. Checkpoint & Kill Job 2 Old (The Workflow You Requested)
        logger.info(f"Signaling Job 2 Old ({job2_old_id}) to checkpoint and exit...")
        
        # Send signal (assuming job2 handles SIGUSR1 or SIGTERM to save)
        DockerLayer.send_signal(job2_old_id, "SIGUSR1") 
        
        # Wait for it to save and die
        Monitor.wait_for_any_exit([job2_old_id])
        logger.info("Job 2 Old has successfully checkpointed and exited.")
        debug_logs(job2_old_id, "job2_old")

        # 4. Start Next Phase
        logger.info(">>> Launching Phase 2 (Job 2 New + Job 3)...")
        job2_new_id = DockerLayer.start_container(IMAGE_NAME, "job2_new", cmd_job2_new, 0, 40, volumes)
        job3_id = DockerLayer.start_container(IMAGE_NAME, "job3", cmd_job3, 0, 60, volumes)
        
        # Optional: Monitor them to completion
        Monitor.wait_for_any_exit([job2_new_id])
        Monitor.wait_for_any_exit([job3_id])
        
        # Dump logs to show resume proof
        debug_logs(job2_new_id, "job2_new")
        debug_logs(job3_id, "job3")
        
        # Cleanup
        DockerLayer.stop_and_remove(job2_new_id)
        DockerLayer.stop_and_remove(job3_id)
        logger.info("Training Workflow Complete.")

    elif args.mode == 'serve':
        # ----------------------------------------------------------------
        # Serve workflow with the REAL inference workload replacing job2.
        #   Phase 1: job1 (toy, 50%) + w2_old (inference, 50%)
        #   Phase 2: w2_new (inference, 40%) + job3 (toy, 60%)
        # We measure the container-swap time between phases.
        # ----------------------------------------------------------------

        # Build the command that runs the inference inside the container
        serve_container_cmd = f"bash -c 'cd {SERVE_WORKDIR} && {SERVE_INFERENCE_CMD}'"
        # Ensure Python stdout is unbuffered so logs appear in docker logs immediately
        serve_envs = {"PYTHONUNBUFFERED": "1"}

        # 1. START INITIAL STATE: Job1 (toy) + W2_old (inference) at 50/50 MPS
        logger.info(">>> Launching Initial Jobs (Job1 + W2 Old)...")
        job1_id = DockerLayer.start_container(IMAGE_NAME, "job1", cmd_job1, 0, 50, volumes)
        w2_old_id = DockerLayer.start_container(
            IMAGE_NAME, "w2_old", serve_container_cmd, 0, 50, SERVE_VOLUMES, envs=serve_envs
        )

        # 2. Wait for Job1 to complete
        logger.info("Waiting for Job 1 to finish...")
        Monitor.wait_for_any_exit([job1_id])
        logger.info("Job 1 finished.")

        # 3. Spawn W2_new and Job3 immediately with revised MPS partitions
        logger.info(">>> Launching Phase 2 (W2 New + Job 3)...")
        w2_new_id = DockerLayer.start_container(
            IMAGE_NAME, "w2_new", serve_container_cmd, 0, 40, SERVE_VOLUMES, envs=serve_envs
        )
        job3_id = DockerLayer.start_container(IMAGE_NAME, "job3", cmd_job3, 0, 60, volumes)

        # 4. Monitor until both new containers are fully running (or have already exited)
        start_time = time.time()
        w2_new_confirmed = False
        job3_confirmed = False
        while True:
            w2_new_running = DockerLayer.is_container_running(w2_new_id)
            job3_running = DockerLayer.is_container_running(job3_id)

            # Once a container is seen running at least once, mark it confirmed
            if w2_new_running:
                w2_new_confirmed = True
            if job3_running:
                job3_confirmed = True

            if w2_new_confirmed and job3_confirmed:
                logger.info("Both W2 (New) and Job 3 are running.")
                break

            # If a container was never seen running and is already gone, it crashed/exited early
            if not w2_new_running and not w2_new_confirmed:
                # Check if container still exists but is not running (i.e. exited)
                try:
                    import docker as _docker
                    c = _docker.from_env().containers.get(w2_new_id)
                    if c.status in ('exited', 'dead'):
                        logger.error(f"W2 New ({w2_new_id}) exited early! Status: {c.status}")
                        debug_logs(w2_new_id, "w2_new")
                        break
                except:
                    logger.error(f"W2 New ({w2_new_id}) container not found!")
                    break

            if not job3_running and not job3_confirmed:
                try:
                    import docker as _docker
                    c = _docker.from_env().containers.get(job3_id)
                    if c.status in ('exited', 'dead'):
                        logger.error(f"Job 3 ({job3_id}) exited early! Status: {c.status}")
                        debug_logs(job3_id, "job3")
                        break
                except:
                    logger.error(f"Job 3 ({job3_id}) container not found!")
                    break

            time.sleep(0.1)

        total_time = time.time() - start_time
        logger.info(f"Phase 2 containers took {total_time:.2f} seconds to start.")

        # 5. Kill W2_old now that replacements are confirmed running
        logger.info(f"Stopping W2 Old ({w2_old_id})...")
        debug_logs(w2_old_id, "w2_old")
        DockerLayer.stop_and_remove(w2_old_id)
        logger.info("W2 Old stopped and removed.")

        # 6. Wait for new containers to produce output, then dump logs
        logger.info("Waiting 10s for new containers to initialize before dumping logs...")
        time.sleep(10)
        debug_logs(w2_new_id, "w2_new")
        debug_logs(job3_id, "job3")

        # 7. Verify W2_old is gone
        if not DockerLayer.is_container_running(w2_old_id):
            logger.info("Serve Workflow Complete. W2_old successfully replaced.")
        else:
            logger.error("W2_old is still running! Something went wrong.")

    elif args.mode == 'serve-gpu-check':
        # ----------------------------------------------------------------
        # Serve workflow with GPU activity probing.
        #   Same as 'serve', but instead of checking container.status == running,
        #   we wait until the workload is actually using the GPU (nvidia-smi)
        #   before killing w2_old.
        #   Phase 1: job1 (toy, 50%) + w2_old (inference, 50%)
        #   Phase 2: w2_new (inference, 40%) + job3 (toy, 60%)
        # ----------------------------------------------------------------

        # Setup the command
        serve_container_cmd = f"bash -c 'cd {SERVE_WORKDIR} && {SERVE_INFERENCE_CMD}'"
        serve_envs = {"PYTHONUNBUFFERED": "1"}

        # 1. START INITIAL STATE: Job1 (toy) + W2_old (inference) at 50/50 MPS
        logger.info(">>> [GPU-Check] Launching Initial Jobs (Job1 + W2 Old)...")
        job1_id = DockerLayer.start_container(IMAGE_NAME, "job1", cmd_job1, 0, 50, volumes)
        w2_old_id = DockerLayer.start_container(
            IMAGE_NAME, "w2_old", serve_container_cmd, 0, 50, SERVE_VOLUMES, envs=serve_envs
        )

        # 2. Wait for Job1 to complete
        logger.info("Waiting for Job 1 to finish...")
        Monitor.wait_for_any_exit([job1_id])
        logger.info("Job 1 finished.")

        # 3. Spawn W2_new and Job3 with revised MPS partitions
        logger.info(">>> [GPU-Check] Launching Phase 2 (W2 New + Job 3)...")
        w2_new_id = DockerLayer.start_container(
            IMAGE_NAME, "w2_new", serve_container_cmd, 0, 40, SERVE_VOLUMES, envs=serve_envs
        )
        job3_id = DockerLayer.start_container(IMAGE_NAME, "job3", cmd_job3, 0, 60, volumes)

        # 4. Wait for W2_new to actually use the GPU before killing W2_old
        start_time = time.time()
        result = Monitor.wait_for_gpu_run(w2_new_id)
        total_time = time.time() - start_time

        if result is None:
            logger.error("W2 New never started using the GPU. Aborting swap.")
            debug_logs(w2_new_id, "w2_new")
        else:
            logger.info(f"W2 New confirmed on GPU after {total_time:.2f}s. Killing W2 Old...")

            # 5. Kill W2_old now that W2_new is confirmed on GPU
            debug_logs(w2_old_id, "w2_old")
            DockerLayer.stop_and_remove(w2_old_id)
            logger.info("W2 Old stopped and removed.")

        # 6. Dump logs from new containers
        debug_logs(w2_new_id, "w2_new")
        debug_logs(job3_id, "job3")

        # 7. Verify W2_old is gone
        if not DockerLayer.is_container_running(w2_old_id):
            logger.info("Serve-GPU-Check Workflow Complete. W2_old successfully replaced.")
        else:
            logger.error("W2_old is still running! Something went wrong.")

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
            name="w2",
            command="bash",          # keeps the container alive
            gpu_idx=0,
            mps_percentage=50,
            volumes=SERVE_VOLUMES,
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

if __name__ == "__main__":
    main()


# def main():
#     state = get_state()
    
#     # Ensure Volumes (Including MPS!)
#     volumes = {
#         CHECKPOINT_HOST_DIR: {'bind': CHECKPOINT_MOUNT_DIR, 'mode': 'rw'},
#         '/tmp/nvidia-mps': {'bind': '/tmp/nvidia-mps', 'mode': 'rw'},
#         HOST_JOBS_DIR: {'bind': CONTAINER_JOBS_DIR, 'mode': 'ro'}
#     }
    
#     # 1. DEFINE COMMANDS
#     cmd_job1 = f"python {CONTAINER_JOBS_DIR}/job1.py"
#     cmd_job2_old = f"python {CONTAINER_JOBS_DIR}/job2.py --save_path {CHECKPOINT_MOUNT_DIR}/job2_ckpt.pt"
#     cmd_job2_new = f"python {CONTAINER_JOBS_DIR}/job2.py --resume_from {CHECKPOINT_MOUNT_DIR}/job2_ckpt.pt"
#     cmd_job3 = f"python {CONTAINER_JOBS_DIR}/job3.py"

#     # 2. START INITIAL STATE
#     logger.info(">>> Launching Initial Jobs...")
    
#     job1_id = DockerLayer.start_container(IMAGE_NAME, "job1", cmd_job1, 0, 50, volumes)
#     job2_old_id = DockerLayer.start_container(IMAGE_NAME, "job2_old", cmd_job2_old, 0, 50, volumes)
    
#     active_containers = [job1_id, job2_old_id]

#     # finished_id = Monitor.wait_for_any_exit(active_containers)
#     # Keep the workflow simple for now and just wait for Job 1 to finish (since Job 2 is infinite loop until checkpointed)
#     job1_id = Monitor.wait_for_any_exit([job1_id])


#     # Start job2_new and job3 immediately after job1 finishes.
#     job2_new_id = DockerLayer.start_container(IMAGE_NAME, "job2_new", cmd_job2_new, 0, 40, volumes)
#     job3_id = DockerLayer.start_container(IMAGE_NAME, "job3", cmd_job3, 0, 60, volumes)

#     start_time = time.time()
#     while True:
#         job2_new_running = DockerLayer.is_container_running(job2_new_id)
#         job3_running = DockerLayer.is_container_running(job3_id)

#         if job2_new_running and job3_running:
#             logger.info("Both Job 2 (New) and Job 3 are running smoothly...")
#             break
        
#         time.sleep(0.1)

#     total_time = time.time() - start_time
#     logger.info(f"Job2 New took {total_time:.2f} seconds to start.")

#     DockerLayer.stop_and_remove(job2_old_id)


    # # 3. DEFINE RELATIONSHIPS (The Generic Logic)
    # # "If Key finishes -> Checkpoint Value"
    # swap_map = {
    #     job1_id: job2_old_id
    # }
    
    # active_containers = [job1_id, job2_old_id]

    # # 4. MONITOR LOOP (Generic)
    # logger.info(f"Monitoring active containers: {active_containers}")
    
    # # Wait for ANY container to exit
    # finished_id = Monitor.wait_for_any_exit(active_containers)
    
    # # 5. HANDLE THE EVENT
    # if finished_id in swap_map:
    #     # A Trigger Job finished!
    #     victim_id = swap_map[finished_id]
    #     logger.info(f"Trigger Job ({finished_id}) finished. Initiating swap on Victim ({victim_id}).")
        
    #     # Debug: Show logs of the finished job
    #     debug_logs(finished_id, "Trigger Job")
    #     if not DEBUG_MODE: DockerLayer.stop_and_remove(finished_id)

    #     # A. Checkpoint the Victim
    #     logger.info("Sending Signal to Victim...")
    #     DockerLayer.send_signal(victim_id, "SIGUSR1")
        
    #     # B. Wait for Victim to Save & Exit
    #     Monitor.wait_for_any_exit([victim_id])
    #     logger.info("Victim has exited.")
        
    #     debug_logs(victim_id, "Victim Job")
    #     if not DEBUG_MODE: DockerLayer.stop_and_remove(victim_id)

    #     # C. Respawn
    #     logger.info(">>> Spawning New Workloads...")
    #     job2_new_id = DockerLayer.start_container(IMAGE_NAME, "job2_new", cmd_job2_new, 0, 40, volumes)
    #     job3_id = DockerLayer.start_container(IMAGE_NAME, "job3", cmd_job3, 0, 40, volumes)
        
    #     # Monitor the new ones
    #     Monitor.wait_for_any_exit([job2_new_id])
    #     Monitor.wait_for_any_exit([job3_id])
        
    # else:
    #     # The Victim died early? Or unknown job?
    #     logger.error(f"Unexpected container {finished_id} finished first! Aborting.")
    #     debug_logs(finished_id, "Failed Job")

    # logger.info("Experiment Complete.")

# if __name__ == "__main__":
#     if not os.path.exists(CHECKPOINT_HOST_DIR):
#         os.makedirs(CHECKPOINT_HOST_DIR)
#     main()