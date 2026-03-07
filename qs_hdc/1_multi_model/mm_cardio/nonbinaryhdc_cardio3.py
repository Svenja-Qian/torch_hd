import torch
import torchhd
import torch.nn as nn
import torchvision
from torchhd.datasets import Cardiotocography3Clases
from tqdm import tqdm
from torch import Tensor
import statistics
import csv
import os
from datetime import datetime
import random

# Function for performing min-max normalization of the input data samples
def create_min_max_normalize(min: Tensor, max: Tensor):
    def normalize(input: Tensor) -> Tensor:
        return torch.nan_to_num((input - min) / (max - min))

    return normalize

# torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 定义要测试的维度
DIMENSIONS_LIST = [1000, 2000, 4000, 6000, 8000, 10000]
BATCH_SIZE = 32

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, f"nonbinary_cardio_3classes_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Load Cardiotocography3Clases dataset
# We use fold 0 for train/test split
train_ds = Cardiotocography3Clases(DATA_DIR, train=True, fold=0, download=True)
test_ds = Cardiotocography3Clases(DATA_DIR, train=False, fold=0, download=True)

# Dynamic feature dimension check
IN_FEATURES = train_ds.data.shape[1]
print(f"Detected Input Features: {IN_FEATURES}")

# Normalize data based on training set statistics
min_val = torch.min(train_ds.data, 0).values
max_val = torch.max(train_ds.data, 0).values
transform = create_min_max_normalize(min_val, max_val)

train_ds.transform = transform
test_ds.transform = transform

train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class Classifier(nn.Module):
    def __init__(self, num_classes, dimensions, in_features, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.num_classes = num_classes
        self.dimensions = dimensions
        self.centroids = None
        
        # 初始化随机投影矩阵，用于将输入特征映射到高维空间
        # Non-binary: Standard random projection (Gaussian)
        self.projection = torchhd.embeddings.Projection(in_features, dimensions)
        self.projection.to(self.device)
        
        # Class centroids (initialized during fit)
        # 类中心向量，将在 fit 过程中计算
        self.centroids = None
        self.is_fitted = False

    def encode(self, x: Tensor) -> Tensor:
        # Flatten input (for tabular data this is identity if already flattened)
        x = x.view(x.size(0), -1)
        
        # 输入数据中心化
        x = x - 0.5
        
        # Project to high-dimensional space
        sample_hv = self.projection(x)
        
        # Non-binary: Return the projected float values directly
        return sample_hv

    def fit(self, data_loader):
        print("Training model...")
        self.train()
        
        # Online accumulation buffer: (num_classes, dimensions)
        # Non-binary: Use float buffer for continuous values
        class_accumulators = torch.zeros(self.num_classes, self.dimensions, dtype=torch.float32, device=self.device)
        
        # Online accumulation of continuous samples
        with torch.no_grad():
            for samples, labels in tqdm(data_loader, desc="Training"):
                samples = samples.to(self.device).float()
                labels = labels.to(self.device)
                
                encoded_samples = self.encode(samples)
                class_accumulators.index_add_(0, labels, encoded_samples)
        
        print("Finalizing training...")
        # Non-binary: Centroids are just the accumulated sums (cosine similarity handles normalization)
        self.centroids = class_accumulators
        return self

    def forward(self, samples):
        # Predict using Cosine similarity with centroids
        return torchhd.cosine_similarity(self.encode(samples), self.centroids)

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

# 存储所有结果
results = []

# 对每个维度进行测试
for dim in tqdm(DIMENSIONS_LIST, desc="Dimensions"):
    print(f"\n=== Testing Dimension: {dim} ===")
    
    accuracies = []
    # Run 100 times with random seeds
    for i in range(100):
        # Generate a random seed
        seed = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(seed)
        print(f"\n--- Run {i+1}/100 (Seed={seed}) ---")
        
        model = Classifier(len(train_ds.classes), dim, IN_FEATURES, device=device)
        model.fit(train_ld)
        
        print("Testing model...")
        acc = model.accuracy(test_ld)
        accuracies.append(acc)
        print(f"Run {i+1} Accuracy: {acc * 100:.3f}%")
    
    avg_acc = statistics.mean(accuracies)
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    std_dev = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    
    print(f"\n>>> Average Accuracy over 100 runs: {avg_acc * 100:.3f}%")
    print(f">>> Min Accuracy: {min_acc * 100:.3f}%")
    print(f">>> Max Accuracy: {max_acc * 100:.3f}%")
    print(f">>> Std Dev: {std_dev * 100:.3f}")
    
    # 存储结果
    results.append({
        "dimension": dim,
        "avg_accuracy": avg_acc,
        "min_accuracy": min_acc,
        "max_accuracy": max_acc,
        "std_dev": std_dev
    })

# 保存结果到CSV文件
with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['Dimension', 'Average Accuracy (%)', 'Min Accuracy (%)', 'Max Accuracy (%)', 'Std Dev (%)'])
    # 写入每个维度的结果
    for result in results:
        writer.writerow([
            result["dimension"],
            f'{result["avg_accuracy"] * 100:.3f}',
            f'{result["min_accuracy"] * 100:.3f}',
            f'{result["max_accuracy"] * 100:.3f}',
            f'{result["std_dev"] * 100:.3f}'
        ])

print(f"\nResults saved to: {OUTPUT_FILE}")
