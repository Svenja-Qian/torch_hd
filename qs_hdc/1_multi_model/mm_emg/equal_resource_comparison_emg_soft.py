import torch
import torchhd
import torch.nn as nn
from torchhd.datasets import EMGHandGestures
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
TOTAL_DIMENSIONS = 6000  # Total dimension budget
NUM_EXPERIMENTS = 100
NUM_ENSEMBLE_MODELS = 6 # Max models needed for seeds
BASELINE_DIM = TOTAL_DIMENSIONS
BATCH_SIZE = 1
INPUT_FEATURES = 1024 # 256 time steps * 4 channels

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Load Data
print("Loading EMG dataset...")
data_dir = os.path.abspath(os.path.join(BASE_DIR, "../../../data")) # Adjust path if needed, or just use "../data" relative to script? 
# baseline_emg.py used: data_dir = os.path.abspath("../data") which is likely relative to mm_emg/
# Let's use a robust path. If mm_emg is in .../multi_model/mm_emg, then ../data is .../multi_model/data? 
# Wait, baseline_emg.py has: data_dir = os.path.abspath("../data")
# I will use "../data" as well to be consistent with baseline_emg.py which is in the same folder.
data_dir = "../data"

def transform(x):
    return x.flatten()

train_ds = EMGHandGestures(data_dir, subjects=[0, 1, 2, 3], download=True, transform=transform)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = EMGHandGestures(data_dir, subjects=[4], download=True, transform=transform)
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
        # Input is already flattened by transform
        # Center input, project, then binarize (BSCTensor)
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
            
            # Collect similarity scores from all models (Soft Voting)
            # Each model output: (batch_size, num_classes)
            batch_similarities = torch.zeros(samples.size(0), NUM_CLASSES, device=device)
            
            for model in models:
                model.eval()
                # model(samples) returns Hamming similarity scores
                batch_similarities += model(samples)
            
            # Select class with highest accumulated similarity
            # This naturally handles ties better than hard voting
            final_preds = torch.argmax(batch_similarities, dim=-1)
            
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

def generate_seeds():
    # Generate enough unique seeds for all experiments
    # Need NUM_EXPERIMENTS * NUM_ENSEMBLE_MODELS seeds for ensemble
    # (Baseline uses the first seed of each group)
    total_seeds_needed = NUM_EXPERIMENTS * NUM_ENSEMBLE_MODELS
    
    # Ensure we have enough range
    all_seeds = random.sample(range(100000), total_seeds_needed) # Increased range for safety
    
    # Split into groups
    groups = [all_seeds[i:i+NUM_ENSEMBLE_MODELS] for i in range(0, total_seeds_needed, NUM_ENSEMBLE_MODELS)]
    return groups

def run_experiment():
    seed_groups = generate_seeds()
    print(f"Generated {len(seed_groups)} seed groups.")
    
    baseline_accuracies = []
    ens3_accuracies = []
    ens6_accuracies = []
    
    for i, seeds in enumerate(seed_groups):
        print(f"\n=== Experiment {i+1}/{NUM_EXPERIMENTS} ===")
        print(f"Seeds: {seeds}")
        
        # --- Baseline Run (1 model, 6000 dim) ---
        baseline_seed = seeds[0] 
        print(f"[Baseline] Running with dim={BASELINE_DIM}, seed={baseline_seed}")
        torch.manual_seed(baseline_seed)
        model = Classifier(NUM_CLASSES, BASELINE_DIM, INPUT_FEATURES, device=device)
        model.fit(train_ld)
        
        preds, labels = get_predictions(model, test_ld)
        acc = accuracy_score(labels, preds)
        baseline_accuracies.append(acc)
        print(f"[Baseline] Accuracy: {acc:.4f}")
        
        # --- Ensemble 3 Run (3 models, 2000 dim each) ---
        ens3_dim = TOTAL_DIMENSIONS // 3
        ens3_seeds = seeds[:3]
        print(f"[Ensemble 3] Running with 3 models, dim per model={ens3_dim}")
        
        models_3 = []
        for seed in ens3_seeds:
            torch.manual_seed(seed)
            sub_model = Classifier(NUM_CLASSES, ens3_dim, INPUT_FEATURES, device=device)
            sub_model.fit(train_ld)
            models_3.append(sub_model)
            
        preds, labels = ensemble_predict(models_3, test_ld)
        acc = accuracy_score(labels, preds)
        ens3_accuracies.append(acc)
        print(f"[Ensemble 3] Accuracy: {acc:.4f}")

        # --- Ensemble 6 Run (6 models, 1000 dim each) ---
        ens6_dim = TOTAL_DIMENSIONS // 6
        ens6_seeds = seeds[:6]
        print(f"[Ensemble 6] Running with 6 models, dim per model={ens6_dim}")
        
        models_6 = []
        for seed in ens6_seeds:
            torch.manual_seed(seed)
            sub_model = Classifier(NUM_CLASSES, ens6_dim, INPUT_FEATURES, device=device)
            sub_model.fit(train_ld)
            models_6.append(sub_model)
            
        preds, labels = ensemble_predict(models_6, test_ld)
        acc = accuracy_score(labels, preds)
        ens6_accuracies.append(acc)
        print(f"[Ensemble 6] Accuracy: {acc:.4f}")

    # --- Results & Reporting ---
    print("\n\n================ RESULTS ================")
    
    mean_base = statistics.mean(baseline_accuracies)
    std_base = statistics.stdev(baseline_accuracies)
    
    mean_ens3 = statistics.mean(ens3_accuracies)
    std_ens3 = statistics.stdev(ens3_accuracies)

    mean_ens6 = statistics.mean(ens6_accuracies)
    std_ens6 = statistics.stdev(ens6_accuracies)
    
    print(f"Baseline (Dim={BASELINE_DIM}): Mean={mean_base:.4f}, Std={std_base:.4f}")
    print(f"Ensemble 3 (Dim={TOTAL_DIMENSIONS//3}): Mean={mean_ens3:.4f}, Std={std_ens3:.4f}")
    print(f"Ensemble 6 (Dim={TOTAL_DIMENSIONS//6}): Mean={mean_ens6:.4f}, Std={std_ens6:.4f}")
    
    # Save CSV
    csv_filename = f"comparison_emg_1_3_6_models_{TOTAL_DIMENSIONS}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Baseline_Acc', 'Ensemble3_Acc', 'Ensemble6_Acc', 'Seeds'])
        for i in range(NUM_EXPERIMENTS):
            writer.writerow([i+1, baseline_accuracies[i], ens3_accuracies[i], ens6_accuracies[i], seed_groups[i]])
        writer.writerow([])
        writer.writerow(['Average', mean_base, mean_ens3, mean_ens6])
        writer.writerow(['StdDev', std_base, std_ens3, std_ens6])
    print(f"Detailed results saved to {csv_path}")

if __name__ == "__main__":
    run_experiment()
