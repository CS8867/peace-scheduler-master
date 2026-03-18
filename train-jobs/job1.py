# jobs/job1.py
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN
import time
import os

def main():
    print("🚀 [JOB 1] Starting High Priority Training...")
    
    # 1. Setup Data (Downloads automatically)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 3. Training Loop (Fixed Duration)
    EPOCHS = 10
    print(f"   [JOB 1] Training for {EPOCHS} epochs on {device}...")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"   [JOB 1] Epoch {epoch + 1}, Batch {i + 1}: Loss {running_loss / 100:.3f}")
                running_loss = 0.0
        
    print("✅ [JOB 1] Finished Training. Exiting.")

if __name__ == "__main__":
    main()