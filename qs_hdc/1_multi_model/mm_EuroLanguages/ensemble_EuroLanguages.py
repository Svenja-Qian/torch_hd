import os
import torch
import torchhd
import torch.nn as nn
from torchhd.datasets import EuropeanLanguages
from tqdm import tqdm
import statistics
import csv
import random
from datetime import datetime

# torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 定义要测试的维度
DIMENSIONS_LIST = [1000, 2000, 4000, 6000, 8000, 10000]
INPUT_FEATURES = 27
BATCH_SIZE = 32

# 路径设置
BASE_DIR = os.path.abspath(".")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, f"ensemble_euro_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load EuropeanLanguages using absolute path
data_dir = os.path.abspath("../data")

def transform(x: str) -> torch.Tensor:
    hist = torch.zeros(INPUT_FEATURES)
    x = x.lower()
    for char in x:
        if 'a' <= char <= 'z':
            hist[ord(char) - ord('a')] += 1
        elif char == ' ':
            hist[26] += 1
    
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist

train_ds = EuropeanLanguages(data_dir, train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = EuropeanLanguages(data_dir, train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class Classifier(nn.Module):
    def __init__(self, num_classes, dimensions, in_features, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.num_classes = num_classes
        self.dimensions = dimensions
        self.centroids = None
        
        self.projection = torchhd.embeddings.Projection(in_features, dimensions)
        with torch.no_grad():
            self.projection.weight.data = self.projection.weight.data.sign()
            self.projection.weight.data[self.projection.weight.data == 0] = 1
        self.projection.to(self.device)
        
        self.centroids = None
        self.is_fitted = False

    def encode(self, x):
        return torchhd.BSCTensor(self.projection(x - 0.5) > 0)

    def fit(self, data_loader):
        print("Training model...")
        self.train()
        class_accumulators = torch.zeros(self.num_classes, self.dimensions, dtype=torch.int32, device=self.device)
        
        with torch.no_grad():
            for samples, labels in tqdm(data_loader, desc="Training"):
                samples = samples.to(self.device).float()
                labels = labels.to(self.device)
                bipolar = torch.where(self.encode(samples), 
                                    torch.tensor(1, device=self.device, dtype=torch.int32), 
                                    torch.tensor(-1, device=self.device, dtype=torch.int32))
                class_accumulators.index_add_(0, labels, bipolar)
        
        print("Finalizing training (Thresholding)...")
        self.centroids = torchhd.BSCTensor(class_accumulators > 0)
        return self

    def forward(self, samples):
        return torchhd.hamming_similarity(self.encode(samples), self.centroids)

    def predict(self, samples):
        return torch.argmax(self(samples), dim=-1)

# 投票预测函数
def vote_predict(models, samples):
    predictions = []
    for model in models:
        pred = model.predict(samples)
        predictions.append(pred)
    
    predictions = torch.stack(predictions)
    batch_size = samples.size(0)
    voted_predictions = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        sample_preds = predictions[:, i]
        counts = torch.bincount(sample_preds)
        voted_predictions[i] = torch.argmax(counts)
    
    return voted_predictions

# 投票模型的准确率计算
def vote_accuracy(models, data_loader):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for samples, labels in tqdm(data_loader, desc="Testing Ensemble"):
            samples = samples.to(device).float()
            labels = labels.to(device)
            predictions = vote_predict(models, samples)
            n_correct += torch.sum(predictions == labels).item()
            n_total += labels.size(0)
    return n_correct / n_total

# 存储所有结果
# ensemble_results = []

# Initialize CSV with header
with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Dimension', 'Average Accuracy (%)', 'Min Accuracy (%)', 'Max Accuracy (%)', 'Std Dev (%)'])

# 对每个维度进行测试
for dim in tqdm(DIMENSIONS_LIST, desc="Dimensions"):
    print(f"\n=== Testing Dimension: {dim} ===")
    
    # 生成10组随机seed，每组3个不重复的seed
    seed_groups = []
    for _ in range(10):
        seeds = random.sample(range(10), 3)
        seed_groups.append(seeds)
    
    accuracies = []
    for i, seeds in enumerate(seed_groups):
        print(f"\n--- Run {i+1}/10 (Seeds={seeds}) ---")
        
        # 训练三个不同seed的模型
        models = []
        for seed in seeds:
            torch.manual_seed(seed)
            print(f"Training model with seed {seed}...")
            model = Classifier(len(train_ds.classes), dim, INPUT_FEATURES, device=device)
            model.fit(train_ld)
            models.append(model)
        
        # 测试投票模型
        print("Testing ensemble model...")
        acc = vote_accuracy(models, test_ld)
        accuracies.append(acc)
        print(f"Run {i+1} Accuracy: {acc * 100:.3f}%")
    
    # 计算统计信息
    avg_acc = statistics.mean(accuracies)
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    std_dev = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    
    print(f"\n>>> Average Accuracy over 10 runs: {avg_acc * 100:.3f}%")
    print(f">>> Min Accuracy: {min_acc * 100:.3f}%")
    print(f">>> Max Accuracy: {max_acc * 100:.3f}%")
    print(f">>> Std Dev: {std_dev * 100:.3f}")
    
    # Save result immediately
    with open(OUTPUT_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            dim,
            f'{avg_acc * 100:.3f}',
            f'{min_acc * 100:.3f}',
            f'{max_acc * 100:.3f}',
            f'{std_dev * 100:.3f}'
        ])

print(f"\nResults saved to: {OUTPUT_FILE}")
