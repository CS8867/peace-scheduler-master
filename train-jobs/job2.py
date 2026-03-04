# jobs/job2.py
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import logging
from model import SimpleCNN
from framework import CheckpointManager  # <--- The only "Framework" touchpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - JOB2 - %(message)s', force=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./checkpoints/job2.pt')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=10000)
    args = parser.parse_args()

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # 2. Initialize Framework (Handles Loading & Signals transparently)
    # The framework sets 'start_epoch' based on whether we are resuming or not.
    manager = CheckpointManager(net, optimizer, args.save_path, args.resume_from)

    # 3. Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

    # 4. Loop
    logging.info(f"Starting Loop from Epoch {manager.start_epoch}...")
    
    for epoch in range(manager.start_epoch, args.max_epochs):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # --- THE FIX: INSTANT CHECK ---
            # Check for signal before every batch (or every N batches)
            manager.check_if_should_exit(current_epoch=epoch, current_loss=running_loss)
            # ------------------------------

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logging.info(f"Finished Epoch {epoch}. Loss: {running_loss/100:.3f}")

if __name__ == "__main__":
    main()