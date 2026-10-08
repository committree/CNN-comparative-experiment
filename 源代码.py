import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

batch_size = 64
epochs = 5
lr = 1e-3

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#基准MLP模型
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

#LeNet风格CNN模型
class CNNLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.con_blocks = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7,128),
            nn.ReLU(),
            nn.Linear(128,10),
        )

    def forward(self, x):
        x = self.con_blocks(x)
        x = x.flatten(start_dim=1)
        return self.fc_layers(x)

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss, epoch_time


def test_model(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    return acc

def run_full_experiment(model, model_name):
    print(f"\n===== 开始训练 {model_name} =====")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} 总参数量: {total_params:,}")

    total_epoch_time = 0.0
    for epoch in range(epochs):
        loss, epoch_time = train(model, device, train_loader, optimizer, criterion)
        train_acc = test_model(model, device, train_loader)
        test_acc = test_model(model, device, test_loader)
        total_epoch_time += epoch_time
        print( f"Epoch {epoch + 1}/{epochs} | 耗时: {epoch_time:.2f}s | 训练损失: {loss:.4f} | 训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")

    avg_epoch_time = total_epoch_time / epochs
    final_test_acc = test_model(model, device, test_loader)
    print(f"{model_name} 训练完成 | 平均每轮耗时: {avg_epoch_time:.2f}s | 最终测试准确率: {final_test_acc:.2f}%")
    return {
        "name": model_name,
        "params": total_params,
        "avg_time_per_epoch": avg_epoch_time,
        "final_acc": final_test_acc,
        "trained_model": model
    }

mlp_result = run_full_experiment(MLPModel(), "MNIST-MLP")
cnn_result = run_full_experiment(CNNLeNet(), "MNIST-CNN-LeNet")