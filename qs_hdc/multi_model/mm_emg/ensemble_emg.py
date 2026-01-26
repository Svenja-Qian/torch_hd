import os
import torch
import torchhd
import torch.nn as nn
from torchhd.datasets import EMGHandGestures
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
INPUT_FEATURES = 1024  # 256 time steps * 4 channels
BATCH_SIZE = 1

# 路径设置
BASE_DIR = os.path.abspath(".")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, f"ensemble_emg_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load EMGHandGestures using absolute path
data_dir = os.path.abspath("../data")

def transform(x):
    return x.flatten()

# Split subjects for training (0-3) and testing (4)
train_ds = EMGHandGestures(data_dir, subjects=[0, 1, 2, 3], download=True, transform=transform)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = EMGHandGestures(data_dir, subjects=[4], download=True, transform=transform)
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

    def encode(self, x):
        # Center input, project, then binarize (BSCTensor)
        # 输入数据中心化 (对 FPGA 友好，且有助于二值化)
        return torchhd.BSCTensor(self.projection(x - 0.5) > 0)

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

# 投票预测函数
def vote_predict(models, samples):
    # 收集所有模型的预测结果
    predictions = []
    for model in models:
        pred = model.predict(samples)
        predictions.append(pred)
    
    # 转换为张量
    predictions = torch.stack(predictions)
    
    # 对每个样本进行投票
    # 对于每个样本，统计每个类别的投票数
    batch_size = samples.size(0)
    voted_predictions = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        # 获取所有模型对第i个样本的预测
        sample_preds = predictions[:, i]
        # 统计每个类别的出现次数
        counts = torch.bincount(sample_preds)
        # 选择出现次数最多的类别
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
            
            # 使用投票预测
            predictions = vote_predict(models, samples)
            
            # 计算正确预测的数量
            n_correct += torch.sum(predictions == labels).item()
            n_total += labels.size(0)
    
    return n_correct / n_total

# 存储所有结果
ensemble_results = []

# 对每个维度进行测试
for dim in tqdm(DIMENSIONS_LIST, desc="Dimensions"):
    print(f"\n=== Testing Dimension: {dim} ===")
    
    # 生成10组随机seed，每组3个不重复的seed（0~9之间）
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
    
    # 存储结果
    ensemble_results.append({
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
    for result in ensemble_results:
        writer.writerow([
            result["dimension"],
            f'{result["avg_accuracy"] * 100:.3f}',
            f'{result["min_accuracy"] * 100:.3f}',
            f'{result["max_accuracy"] * 100:.3f}',
            f'{result["std_dev"] * 100:.3f}'
        ])

print(f"\nResults saved to: {OUTPUT_FILE}")
