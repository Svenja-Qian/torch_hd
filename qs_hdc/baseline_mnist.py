import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from tqdm import tqdm
from typing import Callable, Tuple, Iterable
from torch import Tensor, LongTensor

# 类型定义
DataLoader = Iterable[Tuple[Tensor, LongTensor]] 

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# 超参数设置
DIMENSIONS = 10000
IMG_SIZE = 28
BATCH_SIZE = 8

# 加载 MNIST 数据集
transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 初始化并训练模型
num_classes = len(train_ds.classes)
# model = BaselineMNISTClassifier(num_classes, DIMENSIONS, IMG_SIZE * IMG_SIZE, device=device)

# Classifier 
class Classifier(nn.Module):
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

    def forward(self, samples: Tensor) -> Tensor:
        """
        计算样本属于每个类别的分数（logits）。
        """
        raise NotImplementedError()

    def fit(self, data_loader: DataLoader):
        """
        在提供的数据上训练分类器。
        """
        raise NotImplementedError()

    def predict(self, samples: Tensor) -> LongTensor:
        """
        预测每个样本的类别。
        """
        # 假设 forward 返回的是分数/logits，分数越高越匹配
        return torch.argmax(self(samples), dim=-1)

    def accuracy(self, data_loader: DataLoader) -> float:
        """
        计算在数据集上的准确率。
        """
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            for samples, labels in data_loader:
                samples = samples.to(self.device).float()
                labels = labels.to(self.device)

                predictions = self.predict(samples)
                n_correct += torch.sum(predictions == labels).item()
                n_total += labels.size(0)

        return n_correct / n_total

# 随机投影编码器
class RandomProjectionEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(RandomProjectionEncoder, self).__init__()
        self.bhv_matrix = nn.Parameter(
            torch.empty(out_features, in_features), 
            requires_grad=False
        )
        nn.init.uniform_(self.bhv_matrix, -1, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        # 居中输入数据 (FPGA 友好)
        x = x - 0.5
        
        sample_hv = torch.matmul(x, self.bhv_matrix.t())
        # 严格二进制编码 (0/1)
        # 原始 sign 是 (-1, 0, 1)，这里我们将 > 0 映射为 1，<= 0 映射为 0
        return (sample_hv > 0).float()

# 具体实现的分类器
class BaselineMNISTClassifier(Classifier):
    def __init__(self, num_classes, dimensions, in_features, device=None):
        super().__init__(device=device)
        self.num_classes = num_classes
        self.dimensions = dimensions
        
        # 初始化编码器
        self.encoder = RandomProjectionEncoder(in_features, dimensions)
        self.encoder.to(self.device)
        
        # 初始化类中心
        self.centroids = torch.zeros(num_classes, dimensions, device=self.device)
        self.class_counts = torch.zeros(num_classes, device=self.device)

    def fit(self, data_loader: DataLoader):
        print("Training model...")
        with torch.no_grad():
            for samples, labels in tqdm(data_loader, desc="Training"):
                samples = samples.to(self.device).float()
                labels = labels.to(self.device)
                
                # 编码 (得到 0/1 向量)
                encoded_samples = self.encoder(samples)
                
                # 累加质心 (整数加法)
                for i in range(encoded_samples.size(0)):
                    label = labels[i]
                    self.centroids[label] += encoded_samples[i]
                    self.class_counts[label] += 1
        
        # 二值化质心
        # 使用多数投票阈值 (>= count / 2)
        print("Finalizing training...")
        for c in range(self.num_classes):
            if self.class_counts[c] > 0:
                threshold = self.class_counts[c] / 2
                self.centroids[c] = (self.centroids[c] > threshold).float()
        
        return self

    def forward(self, samples: Tensor) -> Tensor:
        # 编码 (得到 0/1 向量)
        encoded_samples = self.encoder(samples)
        
        # 计算汉明距离 (FPGA 友好: XOR + Popcount)
        # encoded_samples: (batch, dim) -> (batch, 1, dim)
        # centroids: (classes, dim) -> (1, classes, dim)
        # 0/1 向量的汉明距离就是 XOR 结果中 1 的个数 (Sum)
        # torch.abs(a - b) 对于 0/1 向量等价于 XOR
        
        samples_hv = encoded_samples.unsqueeze(1)
        centroids_hv = self.centroids.unsqueeze(0)
        
        # 计算不匹配的元素数量 (汉明距离)
        hamming_distances = torch.sum(torch.abs(samples_hv - centroids_hv), dim=2)  # (batch, classes)
        
        # 因为 predict 使用 argmax，而我们想要最小距离
        # 所以返回负的距离作为分数
        return -hamming_distances


# 训练
model.fit(train_ld)

# 测试
print("Testing model...")
accuracy_value = model.accuracy(test_ld) * 100
print(f"Testing accuracy of {accuracy_value:.3f}%")
