import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
import logging
import sys
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - JOB2 - %(message)s')

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
        )
    def forward(self, x):
        return self.layers(x)

# Global variables to handle state inside the signal handler
model = None
optimizer = None
current_epoch = 0
args = None

def save_checkpoint_handler(signum, frame):
    """
    This function runs INSTANTLY when main.py sends the signal.
    """
    logging.warning(f"!!! RECEIVED SIGNAL {signum} (SIGUSR1) !!!")
    logging.info("Initiating Emergency Checkpoint...")
    
    save_path = args.save_path if args.save_path else "/app/checkpoints/job2_ckpt.pt"
    
    # Save the global state
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    
    logging.info(f"Checkpoint saved to {save_path}. Exiting gracefully.")
    sys.exit(0)

def run_job2():
    global model, optimizer, current_epoch, args
    
    # 1. Register the Signal Handler
    # When this process receives SIGUSR1 (Signal 10), it will run save_checkpoint_handler
    signal.signal(signal.SIGUSR1, save_checkpoint_handler)
    logging.info("Signal Handler for SIGUSR1 registered. Waiting for commands...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Resume Logic
    if args.resume_from and os.path.exists(args.resume_from):
        logging.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
    else:
        logging.info("Starting fresh Job 2")

    # Training Loop
    while True:
        current_epoch += 1
        
        # Heavy work with larger batches
        data = torch.randn(256, 4096, device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        
        if current_epoch % 10 == 0:
            logging.info(f"Epoch {current_epoch} complete. Loss: {loss.item():.4f}, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        time.sleep(0.2) # Shorter sleep to keep GPU busier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='Where to save the checkpoint')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to load')
    args = parser.parse_args()
    
    run_job2()