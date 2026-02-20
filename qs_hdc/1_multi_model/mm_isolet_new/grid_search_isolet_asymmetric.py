import torch
import torchhd
import torch.nn as nn
from torchhd.datasets import ISOLET
from tqdm import tqdm
from torch import Tensor
import statistics
import csv
import os
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Configuration
NUM_EXPERIMENTS = 20  # Run enough experiments to get stable averages
BATCH_SIZE = 1
INPUT_FEATURES = 617
NUM_CLASSES = 26 # ISOLET has 26 classes

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Data
print("Loading ISOLET dataset...")
# Using the same data path assumption as previous scripts
train_ds = ISOLET("../data", train=True, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ds = ISOLET("../data", train=False, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Pre-fetch test labels for efficiency
y_true = []
for _, label in test_ld:
    y_true.append(label.item())
y_true = np.array(y_true)

class Classifier(nn.Module):
    def __init__(self, num_classes, dimensions, in_features, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.num_classes = num_classes
        self.dimensions = dimensions
        
        # Projection
        self.projection = torchhd.embeddings.Projection(in_features, dimensions)
        with torch.no_grad():
            self.projection.weight.data = self.projection.weight.data.sign()
            self.projection.weight.data[self.projection.weight.data == 0] = 1
        self.projection.to(self.device)
        
        self.centroids = None

    def encode(self, x: Tensor) -> torchhd.BSCTensor:
        x = x - 0.5
        sample_hv = self.projection(x)
        return torchhd.BSCTensor(sample_hv > 0)

    def fit(self, data_loader):
        self.train()
        class_accumulators = torch.zeros(self.num_classes, self.dimensions, dtype=torch.int32, device=self.device)
        
        with torch.no_grad():
            for samples, labels in data_loader:
                samples = samples.to(self.device).float()
                labels = labels.to(self.device)
                bipolar = torch.where(self.encode(samples), 
                                    torch.tensor(1, device=self.device, dtype=torch.int32), 
                                    torch.tensor(-1, device=self.device, dtype=torch.int32))
                class_accumulators.index_add_(0, labels, bipolar)
        
        self.centroids = torchhd.BSCTensor(class_accumulators > 0)
        return self

    def forward(self, samples):
        return torchhd.hamming_similarity(self.encode(samples), self.centroids)

def get_similarity_scores(model, data_loader):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for samples, _ in data_loader:
            samples = samples.to(device).float()
            # shape: [1, num_classes]
            scores = model(samples)
            all_scores.append(scores.cpu().numpy())
    return np.concatenate(all_scores, axis=0)

def run_grid_search():
    # Weights to search: 0.5 to 1.0, step 0.01
    # Also include 0.333... for simple average comparison
    w_grid = np.arange(0.5, 1.01, 0.01)
    # Ensure 1.0 is included cleanly
    if not np.isclose(w_grid[-1], 1.0):
        w_grid = np.append(w_grid, 1.0)
        
    special_ws = [1.0/3.0] # Simple average
    
    # Storage for results
    # keys: w values, values: list of accuracies across experiments
    results_grid = {w: [] for w in w_grid}
    results_special = {w: [] for w in special_ws}
    
    print(f"Starting Grid Search with {NUM_EXPERIMENTS} experiments...")
    
    for i in range(NUM_EXPERIMENTS):
        seed_base = random.randint(0, 100000)
        seed_1 = random.randint(0, 100000)
        seed_2 = random.randint(0, 100000)
        
        print(f"Exp {i+1}/{NUM_EXPERIMENTS} | Seeds: {seed_base}, {seed_1}, {seed_2}")
        
        # 1. Train 4000D Model
        torch.manual_seed(seed_base)
        model_4000 = Classifier(NUM_CLASSES, 4000, INPUT_FEATURES, device=device)
        model_4000.fit(train_ld)
        sim_4000 = get_similarity_scores(model_4000, test_ld)
        
        # 2. Train 1000D Model 1
        torch.manual_seed(seed_1)
        model_1000_1 = Classifier(NUM_CLASSES, 1000, INPUT_FEATURES, device=device)
        model_1000_1.fit(train_ld)
        sim_1000_1 = get_similarity_scores(model_1000_1, test_ld)
        
        # 3. Train 1000D Model 2
        torch.manual_seed(seed_2)
        model_1000_2 = Classifier(NUM_CLASSES, 1000, INPUT_FEATURES, device=device)
        model_1000_2.fit(train_ld)
        sim_1000_2 = get_similarity_scores(model_1000_2, test_ld)
        
        # --- Efficient Numpy Grid Search ---
        # Formula: Final_Sim = w * sim_4000 + ((1-w)/2) * (sim_1000_1 + sim_1000_2)
        # Pre-calculate the sum of small models to save ops
        sim_small_sum = sim_1000_1 + sim_1000_2
        
        # Grid weights
        for w in w_grid:
            final_sim = w * sim_4000 + ((1 - w) / 2) * sim_small_sum
            preds = np.argmax(final_sim, axis=1)
            acc = accuracy_score(y_true, preds)
            results_grid[w].append(acc)
            
        # Special weights
        for w in special_ws:
            final_sim = w * sim_4000 + ((1 - w) / 2) * sim_small_sum
            preds = np.argmax(final_sim, axis=1)
            acc = accuracy_score(y_true, preds)
            results_special[w].append(acc)

    # --- Analysis & Plotting ---
    avg_accs = [np.mean(results_grid[w]) for w in w_grid]
    std_accs = [np.std(results_grid[w]) for w in w_grid]
    
    # Find optimal
    best_idx = np.argmax(avg_accs)
    best_w = w_grid[best_idx]
    best_acc = avg_accs[best_idx]
    
    # Baseline stats
    acc_4000_only = np.mean(results_grid[1.0]) if 1.0 in results_grid else 0.0 # Should be in grid
    # If 1.0 wasn't exactly hit due to float, find closest
    if 1.0 not in results_grid:
        closest_1 = min(results_grid.keys(), key=lambda x: abs(x-1.0))
        acc_4000_only = np.mean(results_grid[closest_1])

    acc_simple_avg = np.mean(results_special[special_ws[0]])
    
    print("\n\n================ RESULTS ================")
    print(f"Optimal Weight w: {best_w:.2f}")
    print(f"Optimal Accuracy: {best_acc:.4f}")
    print(f"Baseline (w=1.0, 4000D only): {acc_4000_only:.4f}")
    print(f"Simple Average (w=0.33, Equal): {acc_simple_avg:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(w_grid, avg_accs, marker='o', label='Grid Search Accuracy')
    plt.fill_between(w_grid, 
                     np.array(avg_accs) - np.array(std_accs), 
                     np.array(avg_accs) + np.array(std_accs), 
                     alpha=0.2, label='Std Dev')
    
    # Mark special points
    plt.axvline(x=best_w, color='r', linestyle='--', label=f'Best w={best_w:.2f}')
    plt.axhline(y=acc_simple_avg, color='g', linestyle=':', label=f'Simple Avg (w=0.33): {acc_simple_avg:.4f}')
    plt.axhline(y=acc_4000_only, color='orange', linestyle=':', label=f'4000D Only (w=1.0): {acc_4000_only:.4f}')
    
    plt.title("Weight w vs Accuracy (4000D + 2x1000D Ensemble)")
    plt.xlabel("Weight w (for 4000D model)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, "isolet_asymmetric_grid_search.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Save CSV of grid data
    csv_path = os.path.join(RESULTS_DIR, "isolet_asymmetric_grid_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Weight_w', 'Average_Accuracy', 'Std_Dev'])
        writer.writerow([special_ws[0], acc_simple_avg, np.std(results_special[special_ws[0]])]) # Add simple avg first
        for w, acc, std in zip(w_grid, avg_accs, std_accs):
            writer.writerow([w, acc, std])
    print(f"Grid data saved to {csv_path}")

if __name__ == "__main__":
    run_grid_search()
