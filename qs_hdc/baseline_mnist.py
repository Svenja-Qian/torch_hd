import torch
import torchhd
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch import Tensor

import argparse

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

DIMENSIONS = args.dim
IMG_SIZE = 28
BATCH_SIZE = args.batch_size

# Load MNIST dataset
transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class Classifier(nn.Module):
    def __init__(self, num_classes, dimensions, in_features, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.num_classes = num_classes
        self.dimensions = dimensions
        self.centroids = None
        
        # 初始化随机投影矩阵，用于将输入特征映射到高维空间
        # 硬件友好优化：强制投影矩阵权重为二值 (+1/-1)，避免浮点乘法 (在硬件中可用加减法替代)
        self.projection = torchhd.embeddings.Projection(in_features, dimensions)
        with torch.no_grad():
            self.projection.weight.data = self.projection.weight.data.sign()
            # Fix potential zeros to 1
            self.projection.weight.data[self.projection.weight.data == 0] = 1
        self.projection.to(self.device)
        
        # Class centroids (initialized during fit)
        # 类中心向量，将在 fit 过程中计算
        self.centroids = None
        self.is_fitted = False

    def encode(self, x: Tensor) -> torchhd.BSCTensor:
        # Flatten input
        x = x.view(x.size(0), -1)
        # 输入数据中心化 (对 FPGA 友好，且有助于二值化)
        x = x - 0.5
        
        # Project to high-dimensional space
        sample_hv = self.projection(x)
        # Binarize (threshold > 0)
        # 二值化处理：大于0设为True，否则为False (BSCTensor)
        return torchhd.BSCTensor(sample_hv > 0)

    def fit(self, data_loader):
        print("Training model...")
        self.train()
        
        # Online accumulation buffer: (num_classes, dimensions)
        # Memory Efficiency: Use an integer buffer to count bit occurrences per class.
        # 内存效率优化：使用整数缓冲区进行在线累加，避免存储所有样本的编码向量
        # Initialize with 0
        class_accumulators = torch.zeros(self.num_classes, self.dimensions, dtype=torch.int32, device=self.device)
        
        # Online accumulation of bipolar samples (+1/-1)
        with torch.no_grad():
            for samples, labels in tqdm(data_loader, desc="Training"):
                samples = samples.to(self.device).float()
                labels = labels.to(self.device)
                bipolar = torch.where(self.encode(samples), 
                                    torch.tensor(1, device=self.device, dtype=torch.int32), 
                                    torch.tensor(-1, device=self.device, dtype=torch.int32))
                class_accumulators.index_add_(0, labels, bipolar)
        
        print("Finalizing training (Thresholding)...")
        # Majority Vote: sum > 0 -> 1, sum <= 0 -> 0
        # This creates the binary centroids
        # 多数投票：累加和大于0的维度设为1 (True)，否则为0 (False)
        self.centroids = torchhd.BSCTensor(class_accumulators > 0)
        return self

    def forward(self, samples):
        # Predict using Hamming similarity with centroids
        return torchhd.hamming_similarity(self.encode(samples), self.centroids)

    def predict(self, samples):
        return torch.argmax(self(samples), dim=-1)

    def accuracy(self, data_loader):
        self.eval()
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for samples, labels in data_loader:
                samples = samples.to(self.device).float()
                labels = labels.to(self.device)
                n_correct += torch.sum(self.predict(samples) == labels).item()
                n_total += labels.size(0)
        return n_correct / n_total

model = Classifier(len(train_ds.classes), DIMENSIONS, IMG_SIZE * IMG_SIZE, device=device)
model.fit(train_ld)
print("Testing model...")
print(f"Testing accuracy of {model.accuracy(test_ld) * 100:.3f}%")
