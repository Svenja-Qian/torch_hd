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
TOTAL_DIMENSIONS = 6000
NUM_EXPERIMENTS = 50
BATCH_SIZE = 1
INPUT_FEATURES = 617

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    
    with torch.no_grad():
        for samples, labels in data_loader:
            samples = samples.to(device).float()
            labels = labels.to(device)
            
            # Soft Voting: Accumulate similarity scores
            batch_similarities = torch.zeros(samples.size(0), NUM_CLASSES, device=device)
            
            for model in models:
                model.eval()
                batch_similarities += model(samples)
            
            final_preds = torch.argmax(batch_similarities, dim=-1)
            
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

def generate_seeds():
    # We need seeds for up to 6 models per experiment
    max_models = 6
    num_seeds = NUM_EXPERIMENTS * max_models
    all_seeds = random.sample(range(100000), num_seeds)
    groups = [all_seeds[i:i+max_models] for i in range(0, num_seeds, max_models)]
    return groups

def run_experiment():
    seed_groups = generate_seeds()
    print(f"Generated {len(seed_groups)} seed groups.")
    
    results = {
        "Baseline (1x6000)": [],
        "Sym (6x1000)": [],
        "Asym A (1x4000, 2x1000)": [],
        "Asym B (1x2000, 4x1000)": []
    }
    
    for i, seeds in enumerate(seed_groups):
        print(f"\n=== Experiment {i+1}/{NUM_EXPERIMENTS} ===")
        print(f"Seeds: {seeds}")
        
        # 1. Baseline: 1x6000
        print("[Baseline] 1x6000")
        torch.manual_seed(seeds[0])
        model = Classifier(NUM_CLASSES, 6000, INPUT_FEATURES, device=device)
        model.fit(train_ld)
        preds, labels = get_predictions(model, test_ld)
        acc = accuracy_score(labels, preds)
        results["Baseline (1x6000)"].append(acc)
        print(f"  Acc: {acc:.4f}")
        
        # 2. Sym: 6x1000
        print("[Sym] 6x1000")
        models_sym = []
        for j in range(6):
            torch.manual_seed(seeds[j])
            m = Classifier(NUM_CLASSES, 1000, INPUT_FEATURES, device=device)
            m.fit(train_ld)
            models_sym.append(m)
        preds, _ = ensemble_predict(models_sym, test_ld)
        acc = accuracy_score(labels, preds)
        results["Sym (6x1000)"].append(acc)
        print(f"  Acc: {acc:.4f}")
        
        # 3. Asym A: 1x4000, 2x1000
        print("[Asym A] 1x4000, 2x1000")
        models_asym_a = []
        # Model 1: 4000
        torch.manual_seed(seeds[0])
        m1 = Classifier(NUM_CLASSES, 4000, INPUT_FEATURES, device=device)
        m1.fit(train_ld)
        models_asym_a.append(m1)
        # Models 2-3: 1000
        for j in range(1, 3):
            torch.manual_seed(seeds[j])
            m = Classifier(NUM_CLASSES, 1000, INPUT_FEATURES, device=device)
            m.fit(train_ld)
            models_asym_a.append(m)
        preds, _ = ensemble_predict(models_asym_a, test_ld)
        acc = accuracy_score(labels, preds)
        results["Asym A (1x4000, 2x1000)"].append(acc)
        print(f"  Acc: {acc:.4f}")
        
        # 4. Asym B: 1x2000, 4x1000
        print("[Asym B] 1x2000, 4x1000")
        models_asym_b = []
        # Model 1: 2000
        torch.manual_seed(seeds[0])
        m1 = Classifier(NUM_CLASSES, 2000, INPUT_FEATURES, device=device)
        m1.fit(train_ld)
        models_asym_b.append(m1)
        # Models 2-5: 1000
        for j in range(1, 5):
            torch.manual_seed(seeds[j])
            m = Classifier(NUM_CLASSES, 1000, INPUT_FEATURES, device=device)
            m.fit(train_ld)
            models_asym_b.append(m)
        preds, _ = ensemble_predict(models_asym_b, test_ld)
        acc = accuracy_score(labels, preds)
        results["Asym B (1x2000, 4x1000)"].append(acc)
        print(f"  Acc: {acc:.4f}")

    # --- Reporting ---
    print("\n\n================ RESULTS ================")
    
    # Save CSV
    csv_filename = f"comparison_isolet_architectures_6000.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Experiment', 'Baseline_1x6000', 'Sym_6x1000', 'AsymA_1x4000_2x1000', 'AsymB_1x2000_4x1000', 'Seeds']
        writer.writerow(header)
        
        for i in range(NUM_EXPERIMENTS):
            row = [
                i+1, 
                results["Baseline (1x6000)"][i],
                results["Sym (6x1000)"][i],
                results["Asym A (1x4000, 2x1000)"][i],
                results["Asym B (1x2000, 4x1000)"][i],
                seed_groups[i]
            ]
            writer.writerow(row)
            
        # Stats row
        avgs = [np.mean(results[k]) for k in results]
        stds = [np.std(results[k]) for k in results]
        
        writer.writerow([])
        writer.writerow(['Average', *avgs])
        writer.writerow(['StdDev', *stds])

    print(f"Saved results to {csv_path}")
    
    # Print summary
    keys = list(results.keys())
    for i, k in enumerate(keys):
        print(f"{k}: Mean={avgs[i]:.4f}, Std={stds[i]:.4f}")

if __name__ == "__main__":
    run_experiment()
