import os
import torch
import torchhd
import torch.nn as nn
from torchhd.datasets import EMGHandGestures
from tqdm import tqdm
import statistics

import argparse

# torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

DIMENSIONS = args.dim
INPUT_FEATURES = 1024  # 256 time steps * 4 channels
BATCH_SIZE = args.batch_size

# Load EMGHandGestures using absolute path
data_dir = os.path.abspath("data")

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
        
        # Binary projection (+1/-1) for hardware efficiency
        self.projection = torchhd.embeddings.Projection(in_features, dimensions)
        with torch.no_grad():
            self.projection.weight.data = self.projection.weight.data.sign()
            self.projection.weight.data[self.projection.weight.data == 0] = 1
        self.projection.to(self.device)

    def encode(self, x):
        # Center input, project, then binarize (BSCTensor)
        return torchhd.BSCTensor(self.projection(x - 0.5) > 0)

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

accuracies = []
for i in range(10):
    torch.manual_seed(i)
    print(f"\n--- Run {i+1}/10 (Seed={i}) ---")
    
    model = Classifier(len(train_ds.classes), DIMENSIONS, INPUT_FEATURES, device=device)
    model.fit(train_ld)
    
    print("Testing model...")
    acc = model.accuracy(test_ld)
    accuracies.append(acc)
    print(f"Run {i+1} Accuracy: {acc * 100:.3f}%")

avg_acc = statistics.mean(accuracies)
std_dev = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0

print(f"\n>>> Average Accuracy over 10 runs: {avg_acc * 100:.3f}%")
print(f">>> Std Dev: {std_dev * 100:.3f}")
