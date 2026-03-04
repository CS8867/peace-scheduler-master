import torch
import time
import logging

# FORCE LOGGING TO STDOUT
logging.basicConfig(level=logging.INFO, format='%(asctime)s - JOB1 - %(message)s', force=True)

def run_job1():
    logging.info("Starting Job 1 (Heavy GPU Workload)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run for 60 seconds with intensive GPU work
    start_time = time.time()
    steps = 0
    
    while time.time() - start_time < 60:
        # Large matrix multiplications to saturate GPU
        a = torch.randn(8192, 8192, device=device)
        b = torch.randn(8192, 8192, device=device)
        c = torch.matmul(a, b)
        # Chain more ops to keep GPU busy
        d = torch.matmul(c, a)
        e = torch.svd(c[:1024, :1024])  # SVD on a submatrix
        torch.cuda.synchronize()
        
        steps += 1
        if steps % 2 == 0:
            logging.info(f"Step {steps}: Job 1 running... ({int(time.time() - start_time)}s elapsed, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB)")

    logging.info("Job 1 Finished. Exiting.")

if __name__ == "__main__":
    run_job1()