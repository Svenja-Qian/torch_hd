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
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Configuration
NUM_EXPERIMENTS = 100
NUM_ENSEMBLE_MODELS = 3
BATCH_SIZE = 1
INPUT_FEATURES = 617

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Load Data
print("Loading ISOLET dataset...")
train_ds = ISOLET("../data", train=True, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ds = ISOLET("../data", train=False, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_ds.classes)
print(f"Classes: {NUM_CLASSES}")

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

    def predict(self, samples):
        return torch.argmax(self(samples), dim=-1)

def get_predictions(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for samples, labels in data_loader:
            samples = samples.to(device).float()
            labels = labels.to(device)
            preds = model.predict(samples)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

def ensemble_predict(models, data_loader):
    all_preds = []
    all_labels = []
    
    # Pre-load data to avoid re-fetching
    # But data_loader is a generator, so we iterate normally.
    # To save time, we can iterate once and feed to all models, or just iterate inside.
    # Iterating inside is cleaner.
    
    with torch.no_grad():
        for samples, labels in data_loader:
            samples = samples.to(device).float()
            labels = labels.to(device)
            
            # Collect predictions from all models
            batch_votes = []
            for model in models:
                model.eval()
                # model.predict returns (batch_size,) indices
                batch_votes.append(model.predict(samples))
            
            # Stack: (num_models, batch_size)
            batch_votes = torch.stack(batch_votes)
            
            # Majority Vote
            # Transpose to (batch_size, num_models)
            batch_votes = batch_votes.t()
            
            final_preds = []
            for i in range(batch_votes.size(0)):
                # bincount requires 1D
                counts = torch.bincount(batch_votes[i], minlength=NUM_CLASSES)
                final_preds.append(torch.argmax(counts).item())
            
            all_preds.extend(final_preds)
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

def generate_seeds():
    num_seeds = NUM_EXPERIMENTS * NUM_ENSEMBLE_MODELS
    # Ensure enough unique seeds
    all_seeds = random.sample(range(100000), num_seeds)
    # Split into groups
    groups = [all_seeds[i:i+NUM_ENSEMBLE_MODELS] for i in range(0, num_seeds, NUM_ENSEMBLE_MODELS)]
    return groups

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

def print_per_class_accuracy(cm, classes):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        # ISOLET classes are letters, but let's just use index if class names not available
        # train_ds.classes usually gives names
        class_name = classes[i] if i < len(classes) else str(i)
        print(f"  Class {class_name}: {acc:.4f}")
    
    # Identify similar classes (where confusion is high)
    # This is a bit complex to automate perfectly, but we can print low accuracy classes
    sorted_indices = np.argsort(per_class_acc)
    print("\n  Top 5 Hardest Classes:")
    for i in sorted_indices[:5]:
        class_name = classes[i] if i < len(classes) else str(i)
        print(f"    Class {class_name}: {per_class_acc[i]:.4f}")

def run_experiment(total_dims):
    baseline_dim = total_dims
    ensemble_dim = total_dims // NUM_ENSEMBLE_MODELS
    
    seed_groups = generate_seeds()
    print(f"Generated {len(seed_groups)} seed groups for Total Dim: {total_dims}.")
    
    baseline_accuracies = []
    baseline_cms = []
    
    ensemble_accuracies = []
    ensemble_cms = []
    
    for i, seeds in enumerate(seed_groups):
        print(f"\n=== Experiment {i+1}/{NUM_EXPERIMENTS} ===")
        print(f"Seeds: {seeds}")
        
        # --- Baseline Run ---
        baseline_seed = seeds[0] # Use first seed for baseline
        print(f"[Baseline] Running with dim={baseline_dim}, seed={baseline_seed}")
        torch.manual_seed(baseline_seed)
        model = Classifier(NUM_CLASSES, baseline_dim, INPUT_FEATURES, device=device)
        model.fit(train_ld)
        
        preds, labels = get_predictions(model, test_ld)
        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))
        
        baseline_accuracies.append(acc)
        baseline_cms.append(cm)
        print(f"[Baseline] Accuracy: {acc:.4f}")
        
        # --- Ensemble Run ---
        print(f"[Ensemble] Running with {NUM_ENSEMBLE_MODELS} models, dim per model={ensemble_dim}")
        models = []
        for seed in seeds:
            torch.manual_seed(seed)
            sub_model = Classifier(NUM_CLASSES, ensemble_dim, INPUT_FEATURES, device=device)
            sub_model.fit(train_ld)
            models.append(sub_model)
            
        preds, labels = ensemble_predict(models, test_ld)
        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))
        
        ensemble_accuracies.append(acc)
        ensemble_cms.append(cm)
        print(f"[Ensemble] Accuracy: {acc:.4f}")

    # --- Results & Reporting ---
    print("\n\n================ RESULTS ================")
    
    # Baseline Statistics
    mean_base = statistics.mean(baseline_accuracies)
    std_base = statistics.stdev(baseline_accuracies)
    total_base_cm = sum(baseline_cms)
    
    print(f"Baseline (Dim={baseline_dim}):")
    print(f"  Mean Accuracy: {mean_base:.4f}")
    print(f"  Std Dev:       {std_base:.4f}")
    print_per_class_accuracy(total_base_cm, train_ds.classes)
    
    # Ensemble Statistics
    mean_ens = statistics.mean(ensemble_accuracies)
    std_ens = statistics.stdev(ensemble_accuracies)
    total_ens_cm = sum(ensemble_cms)
    
    print(f"\nEnsemble ({NUM_ENSEMBLE_MODELS} models, Dim={ensemble_dim} each):")
    print(f"  Mean Accuracy: {mean_ens:.4f}")
    print(f"  Std Dev:       {std_ens:.4f}")
    print_per_class_accuracy(total_ens_cm, train_ds.classes)
    
    # Save Plots
    plot_confusion_matrix(total_base_cm, f"Baseline Confusion Matrix (Acc: {mean_base:.4f})", 
                          os.path.join(RESULTS_DIR, f"isolet_baseline_cm_{total_dims}_{NUM_ENSEMBLE_MODELS}.png"))
    plot_confusion_matrix(total_ens_cm, f"Ensemble Confusion Matrix (Acc: {mean_ens:.4f})", 
                          os.path.join(RESULTS_DIR, f"isolet_ensemble_cm_{total_dims}_{NUM_ENSEMBLE_MODELS}.png"))
    print(f"\nPlots saved to {RESULTS_DIR}")
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"e_r_c_{total_dims}_{NUM_ENSEMBLE_MODELS}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Baseline_Acc', 'Ensemble_Acc', 'Seeds'])
        for i in range(NUM_EXPERIMENTS):
            writer.writerow([i+1, baseline_accuracies[i], ensemble_accuracies[i], seed_groups[i]])
        writer.writerow([])
        writer.writerow(['Average', mean_base, mean_ens])
        writer.writerow(['StdDev', std_base, std_ens])
    print(f"Detailed results saved to {csv_path}")

if __name__ == "__main__":
    for dim in [6000, 12000]:
        run_experiment(dim)
