# jobs/job3.py
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN
import time

def main():
    print("🚀 [JOB 3] Starting Backfill Training Job...")
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # Higher LR just to be different
    criterion = torch.nn.CrossEntropyLoss()

    # 2. Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # 3. Training Loop
    EPOCHS = 2
    print(f"   [JOB 3] Training for {EPOCHS} epochs...")

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
                print(f"   [JOB 3] Epoch {epoch + 1}, Batch {i + 1}: Loss {running_loss / 100:.3f}")
                running_loss = 0.0
                
    print("✅ [JOB 3] Finished Training. Workflow Complete.")

if __name__ == "__main__":
    main()