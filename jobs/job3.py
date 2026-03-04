import torch
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - JOB3 - %(message)s')

def run_job3():
    logging.info("Starting Job 3 (Heavy GPU Workload)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    step = 0
    
    while True:
        # Intensive GPU work: large matmuls + convolutions
        x = torch.randn(8192, 8192, device=device)
        y = torch.matmul(x, x)
        z = torch.matmul(y, x)
        # Additional work: eigenvalue decomposition on submatrix
        eigvals = torch.linalg.eigvalsh(y[:512, :512])
        torch.cuda.synchronize()
        
        step += 1
        if step % 3 == 0:
            logging.info(f"Job 3 running step {step}, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

if __name__ == "__main__":
    run_job3()