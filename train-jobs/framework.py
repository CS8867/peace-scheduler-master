# jobs/framework.py
import signal
import sys
import os
import torch

class CheckpointManager:
    def __init__(self, model, optimizer, save_path, resume_path=None):
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.stop_requested = False
        
        # Resume Logic
        self.start_epoch = 0
        if resume_path and os.path.exists(resume_path):
            print(f"📦 [FRAMEWORK] Found checkpoint at {resume_path}. Loading...")
            ckpt = torch.load(resume_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # If we were interrupted in Epoch 5, we restart Epoch 5.
            self.start_epoch = ckpt['epoch'] 
            print(f"📦 [FRAMEWORK] Resuming from start of Epoch {self.start_epoch}")

        # Signal Registration
        signal.signal(signal.SIGUSR1, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n⚠️ [FRAMEWORK] Signal {signum} received! Flagging for stop...")
        self.stop_requested = True

    def save_and_exit(self, current_epoch, current_loss):
        """Forces a save and kills the process immediately."""
        print(f"💾 [FRAMEWORK] Saving partial state (Epoch {current_epoch}) to {self.save_path}...")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        torch.save({
            'epoch': current_epoch, # Save current epoch so we resume here
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': current_loss,
        }, self.save_path)
        
        print("🛑 [FRAMEWORK] State saved. Exiting process now.")
        sys.exit(0)

    def check_if_should_exit(self, current_epoch, current_loss=0.0):
        """Call this inside the batch loop for instant response."""
        if self.stop_requested:
            self.save_and_exit(current_epoch, current_loss)