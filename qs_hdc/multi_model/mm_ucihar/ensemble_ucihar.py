import os
import torch
import torchhd
import torch.nn as nn
from torchhd.datasets import UCIHAR
from tqdm import tqdm
import statistics
import csv
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 全局参数
INPUT_FEATURES = 561
BATCH_SIZE = 1
NUM_MODELS = 3  # 集成模型数量

# 固定维度设置
FIXED_DIMENSIONS = [10000, 10000, 10000]  # 每个集成包含的三个固定维度

# 路径设置
BASE_DIR = os.path.abspath(".")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 生成结果文件名
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = os.path.join(RESULTS_DIR, f"seed_experiment_results_{TIMESTAMP}.csv")

# 加载数据集
data_dir = os.path.join(BASE_DIR, "data")
train_ds = UCIHAR(data_dir, train=True, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ds = UCIHAR(data_dir, train=False, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class Classifier(nn.Module):
    def __init__(self, num_classes, dimensions, in_features, device=None, center_value=0.5, threshold=0.0):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.num_classes = num_classes
        self.dimensions = dimensions
        self.center_value = center_value  # 中心值
        self.threshold = threshold  # 二值化阈值
        self.centroids = None
        
        # Binary projection (+1/-1) for hardware efficiency
        self.projection = torchhd.embeddings.Projection(in_features, dimensions)
        with torch.no_grad():
            self.projection.weight.data = self.projection.weight.data.sign()
            self.projection.weight.data[self.projection.weight.data == 0] = 1
        self.projection.to(self.device)

    def encode(self, x):
        # Center input, project, then binarize (BSCTensor)
        return torchhd.BSCTensor(self.projection(x - self.center_value) > self.threshold)

    def fit(self, data_loader):
        print("Training model...")
        self.train()
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
        
        self.centroids = torchhd.BSCTensor(class_accumulators > 0)
        return self

    def forward(self, samples):
        # Predict using Hamming similarity with centroids
        return torchhd.hamming_similarity(self.encode(samples), self.centroids)

    def predict(self, samples):
        return torch.argmax(self(samples), dim=-1)

    # def accuracy(self, data_loader):
    #     self.eval()
    #     n_correct = 0
    #     n_total = 0
    #     with torch.no_grad():
    #         for samples, labels in data_loader:
    #             samples = samples.to(self.device).float()
    #             labels = labels.to(self.device)
    #             n_correct += torch.sum(self.predict(samples) == labels).item()
    #             n_total += labels.size(0)
    #     return n_correct / n_total

    # def save(self, path):
    #     """保存模型参数"""
    #     torch.save({
    #         'centroids': self.centroids,
    #         'projection_weight': self.projection.weight,
    #         'center_value': self.center_value,
    #         'threshold': self.threshold
    #     }, path)

    # def load(self, path):
    #     """加载模型参数"""
    #     checkpoint = torch.load(path)
    #     self.centroids = checkpoint['centroids']
    #     self.projection.weight.data = checkpoint['projection_weight']
    #     if 'center_value' in checkpoint:
    #         self.center_value = checkpoint['center_value']
    #     if 'threshold' in checkpoint:
    #         self.threshold = checkpoint['threshold']
    #     return self

def hard_voting_ensemble(models, data_loader):
    """硬投票集成推理"""
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for samples, labels in data_loader:
            samples = samples.to(device).float()
            labels = labels.to(device)
            
            # 收集所有模型的预测
            preds = []
            for model in models:
                model_pred = model.predict(samples)
                preds.append(model_pred)
            
            # 多数投票
            ensemble_pred = torch.mode(torch.stack(preds), dim=0).values
            all_predictions.append(ensemble_pred)
            all_labels.append(labels)
    
    # 计算集成模型的准确率
    ensemble_preds = torch.cat(all_predictions)
    ensemble_acc = torch.sum(ensemble_preds == torch.cat(all_labels)).item() / torch.cat(all_labels).size(0)
    
    return ensemble_acc

def main():
    print(f"Starting dimension experiment for UCIHAR dataset")
    print(f"Number of models per ensemble: {NUM_MODELS}")
    print(f"Number of seed groups per dimension: 10")
    
    # 生成10组随机seed，每组3个不重复的seed（0~9之间）
    import random
    seed_groups = []
    for _ in range(10):
        # 随机生成3个不重复的seed（0~9）
        seeds = random.sample(range(10), 3)
        seed_groups.append(seeds)
    
    print(f"\nGenerated seed groups (showing first 10):")
    for i, seeds in enumerate(seed_groups[:10]):
        print(f"  Group {i+1}: {seeds}")
    print(f"  ... and {len(seed_groups) - 10} more groups")
    
    # 定义维度范围：1000, 2000, 4000, 6000, 8000, 10000
    dimensions = [1000, 2000, 4000, 6000, 8000, 10000]
    print(f"\nTesting dimensions: {dimensions}")
    
    # 存储所有实验结果
    experiment_results = []
    
    # 存储每个维度的统计结果
    dimension_stats = []
    
    # 运行每个维度的实验
    print("\n--- Running Dimension Experiments ---")
    for dim in tqdm(dimensions, desc="Dimensions"):
        print(f"\n=== Testing Dimension: {dim} ===")
        
        # 设置当前维度（三个模型使用相同维度）
        current_dimensions = [dim, dim, dim]
        print(f"Using dimensions: {current_dimensions}")
        
        # 存储当前维度的所有实验结果
        dim_results = []
        
        # 运行10组实验
        for i, seed_group in enumerate(tqdm(seed_groups, desc=f"Seed Groups (Dim: {dim})")):
            if (i + 1) % 10 == 0:
                print(f"  Group {i+1}/{len(seed_groups)} - Seeds: {seed_group}")
            
            # 训练模型
            models = []
            for j in range(NUM_MODELS):
                seed = seed_group[j]
                
                # 设置随机种子
                torch.manual_seed(seed)
                
                # 创建模型
                model = Classifier(len(train_ds.classes), current_dimensions[j], INPUT_FEATURES, 
                                  device=device, center_value=0.5, 
                                  threshold=0.0)
                
                # 训练模型
                model.fit(train_ld)
                models.append(model)
            
            # 测试集成模型
            ensemble_acc = hard_voting_ensemble(models, test_ld)
            
            # 记录结果
            dim_results.append(ensemble_acc)
            experiment_results.append({
                "dimension": dim,
                "group": i+1,
                "seeds": seed_group,
                "ensemble_accuracy": ensemble_acc
            })
        
        # 计算当前维度的统计量
        avg_accuracy = statistics.mean(dim_results)
        min_accuracy = min(dim_results)
        max_accuracy = max(dim_results)
        
        # 存储统计结果
        dimension_stats.append({
            "dimension": dim,
            "avg_accuracy": avg_accuracy,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy
        })
        
        # 打印当前维度的结果
        print(f"\n--- Results for Dimension {dim} ---")
        print(f"Mean Ensemble Accuracy: {avg_accuracy * 100:.3f}%")
        print(f"Min Ensemble Accuracy: {min_accuracy * 100:.3f}%")
        print(f"Max Ensemble Accuracy: {max_accuracy * 100:.3f}%")
    
    # 保存结果到CSV文件
    # 修改结果文件名，添加dimension前缀
    RESULTS_FILE = os.path.join(RESULTS_DIR, f"dimension_experiment_results_{TIMESTAMP}.csv")
    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['Dimension', 'Experiment Group', 'Seed 1', 'Seed 2', 'Seed 3', 'Ensemble Accuracy (%)'])
        # 写入每个实验的结果
        for result in experiment_results:
            writer.writerow([
                result["dimension"],
                result["group"],
                result["seeds"][0],
                result["seeds"][1],
                result["seeds"][2],
                f'{result["ensemble_accuracy"] * 100:.3f}'
            ])
        # 写入空行
        writer.writerow([])
        # 写入统计结果表头
        writer.writerow(['Dimension', 'Average Accuracy (%)', 'Min Accuracy (%)', 'Max Accuracy (%)'])
        # 写入每个维度的统计结果
        for stat in dimension_stats:
            writer.writerow([
                stat["dimension"],
                f'{stat["avg_accuracy"] * 100:.3f}',
                f'{stat["min_accuracy"] * 100:.3f}',
                f'{stat["max_accuracy"] * 100:.3f}'
            ])
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    # 使用matplotlib可视化结果
    print("\n--- Visualizing Results ---")
    
    # 提取维度和平均准确率
    dims = [stat["dimension"] for stat in dimension_stats]
    avg_accs = [stat["avg_accuracy"] * 100 for stat in dimension_stats]
    min_accs = [stat["min_accuracy"] * 100 for stat in dimension_stats]
    max_accs = [stat["max_accuracy"] * 100 for stat in dimension_stats]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制平均准确率曲线
    plt.plot(dims, avg_accs, 'o-', label='Average Ensemble Accuracy', linewidth=2, markersize=8)
    
    # 绘制准确率范围（阴影区域）
    plt.fill_between(dims, min_accs, max_accs, alpha=0.2, label='Accuracy Range')
    
    # 添加标题和标签
    plt.title('Ensemble Accuracy vs. Dimension Size', fontsize=16)
    plt.xlabel('Dimension Size', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 设置x轴刻度
    plt.xticks(dims)
    
    # 保存图表
    plot_file = os.path.join(RESULTS_DIR, f"dimension_accuracy_plot_{TIMESTAMP}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()
