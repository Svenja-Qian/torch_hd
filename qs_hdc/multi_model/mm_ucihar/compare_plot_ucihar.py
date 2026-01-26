import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

# 路径设置
BASE_DIR = os.path.abspath(".")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_IMAGE = os.path.join(RESULTS_DIR, f"ucihar_comparison_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 手动指定CSV文件路径（根据实际生成的文件名修改）
# 请在运行完 baseline_ucihar.py 和 ensemble_ucihar.py 后，将生成的文件名填入此处
baseline_csv = os.path.join(RESULTS_DIR, "baseline_ucihar_results_TIMESTAMP.csv")
ensemble_csv = os.path.join(RESULTS_DIR, "ensemble_ucihar_results_TIMESTAMP.csv")

# 为了方便，如果找不到文件，尝试自动查找最新的文件（可选）
def get_latest_csv(prefix):
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(prefix) and f.endswith('.csv')]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(RESULTS_DIR, files[0])

found_baseline = get_latest_csv("baseline_ucihar_results_")
found_ensemble = get_latest_csv("ensemble_ucihar_results_")

if found_baseline:
    print(f"Found latest baseline CSV: {found_baseline}")
    baseline_csv = found_baseline
if found_ensemble:
    print(f"Found latest ensemble CSV: {found_ensemble}")
    ensemble_csv = found_ensemble

# 检查文件是否存在
if not os.path.exists(baseline_csv) or not os.path.exists(ensemble_csv):
    print(f"Error: CSV files not found.")
    print(f"Baseline: {baseline_csv}")
    print(f"Ensemble: {ensemble_csv}")
    print("Please run baseline_ucihar.py and ensemble_ucihar.py first, or update the file paths in this script.")
    exit(1)

# 读取基准模型结果
baseline_data = {}
with open(baseline_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dim = int(row['Dimension'])
        baseline_data[dim] = {
            'avg': float(row['Average Accuracy (%)']),
            'min': float(row['Min Accuracy (%)']),
            'max': float(row['Max Accuracy (%)'])
        }

# 读取投票模型结果
ensemble_data = {}
with open(ensemble_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        dim = int(row['Dimension'])
        ensemble_data[dim] = {
            'avg': float(row['Average Accuracy (%)']),
            'min': float(row['Min Accuracy (%)']),
            'max': float(row['Max Accuracy (%)'])
        }

# 提取维度列表（确保顺序一致）
dimensions = sorted(baseline_data.keys())

# 提取数据点
baseline_avg = [baseline_data[dim]['avg'] for dim in dimensions]
baseline_min = [baseline_data[dim]['min'] for dim in dimensions]
baseline_max = [baseline_data[dim]['max'] for dim in dimensions]

ensemble_avg = [ensemble_data[dim]['avg'] for dim in dimensions]
ensemble_min = [ensemble_data[dim]['min'] for dim in dimensions]
ensemble_max = [ensemble_data[dim]['max'] for dim in dimensions]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制基准模型的曲线和范围
plt.plot(dimensions, baseline_avg, 's-', label='Baseline Model (Single)', linewidth=2, markersize=8, color='red')
plt.fill_between(dimensions, baseline_min, baseline_max, alpha=0.2, color='red')

# 绘制投票模型的曲线和范围
plt.plot(dimensions, ensemble_avg, 'o-', label='Ensemble Model (3-Vote)', linewidth=2, markersize=8, color='blue')
plt.fill_between(dimensions, ensemble_min, ensemble_max, alpha=0.2, color='blue')

# 设置图表属性
plt.xlabel('Dimension', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('UCIHAR: Baseline vs Ensemble Model Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# 调整x轴和y轴范围
plt.xlim([min(dimensions) - 500, max(dimensions) + 500])
# 自动计算y轴范围，确保所有数据点都能显示
all_accuracies = baseline_min + baseline_max + ensemble_min + ensemble_max
plt.ylim([min(all_accuracies) - 2, max(all_accuracies) + 2])

# 保存图像
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)

print(f"Comparison plot saved to: {OUTPUT_IMAGE}")

# 显示图像（可选）
# plt.show()
