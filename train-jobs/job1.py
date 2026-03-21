# jobs/job1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os


class HeavyCNN(nn.Module):
    """A deeper, wider CNN to make training significantly slower."""
    def __init__(self):
        super(HeavyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    import sys
    print("🚀 [JOB 1] Starting High Priority Training (Heavy Model)...", flush=True)
    job_start = time.time()

    # 1. Setup Data with heavier augmentation
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Smaller batch size = more iterations per epoch = slower
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    # 2. Setup Heavy Model (VGG-style, ~10x parameters of SimpleCNN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = HeavyCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    param_count = sum(p.numel() for p in net.parameters())
    print(f"   [JOB 1] Model has {param_count:,} parameters", flush=True)
    print(f"   [JOB 1] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"   [JOB 1] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"   [JOB 1] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    # 3. Training Loop
    EPOCHS = 2
    print(f"   [JOB 1] Training for {EPOCHS} epochs on {device} (batch_size=16)...", flush=True)
    print(f"   [JOB 1] Total batches per epoch: {len(trainloader)}", flush=True)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 200 == 199:
                elapsed = time.time() - epoch_start
                acc = 100.0 * correct / total
                print(f"   [JOB 1] Epoch {epoch + 1}, Batch {i + 1}/{len(trainloader)}: Loss {running_loss / 200:.3f}, Acc {acc:.1f}%, Time {elapsed:.1f}s", flush=True)
                running_loss = 0.0

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_acc = 100.0 * correct / total
        print(f"   [JOB 1] Epoch {epoch + 1} complete — Acc: {epoch_acc:.1f}%, Epoch Time: {epoch_time:.1f}s", flush=True)

    total_time = time.time() - job_start
    print(f"✅ [JOB 1] Finished Training. Total time: {total_time:.1f}s. Exiting.", flush=True)

if __name__ == "__main__":
    main()