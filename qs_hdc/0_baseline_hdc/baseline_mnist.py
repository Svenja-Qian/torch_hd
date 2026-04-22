import torch
import torchhd
import torch.nn as nn
import gzip
import os
import struct
from pathlib import Path

import requests
from tqdm import tqdm
from torch import Tensor
import statistics

import argparse

# torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=4000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--no_center", action="store_true", help="Disable input centering (x = x - 0.5) before projection.")
parser.add_argument("--data_root", type=str, default="../data")
args = parser.parse_args()

DIMENSIONS = args.dim
IMG_SIZE = 28
BATCH_SIZE = args.batch_size
RUNS = args.runs
DATA_ROOT = args.data_root

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _read_idx_images_gz(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number for images: {magic}")
        data = f.read()
    x = torch.frombuffer(data, dtype=torch.uint8).clone()
    x = x.view(n, rows, cols)
    return x


def _read_idx_labels_gz(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number for labels: {magic}")
        data = f.read()
    y = torch.frombuffer(data, dtype=torch.uint8).clone()
    y = y.view(n)
    return y


def _load_mnist_split(root: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"mnist_{split}.pt"
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        return obj["images"], obj["labels"]

    if split == "train":
        images_name = MNIST_FILES["train_images"]
        labels_name = MNIST_FILES["train_labels"]
    elif split == "test":
        images_name = MNIST_FILES["test_images"]
        labels_name = MNIST_FILES["test_labels"]
    else:
        raise ValueError(f"Unknown split: {split}")

    images_path = root / images_name
    labels_path = root / labels_name

    if not images_path.exists():
        _download(MNIST_BASE_URL + images_name, images_path)
    if not labels_path.exists():
        _download(MNIST_BASE_URL + labels_name, labels_path)

    images = _read_idx_images_gz(images_path)
    labels = _read_idx_labels_gz(labels_path).to(torch.long)

    torch.save({"images": images, "labels": labels}, cache_path)
    return images, labels


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool):
        self.root = Path(root).expanduser().resolve() / "mnist"
        split = "train" if train else "test"
        images, labels = _load_mnist_split(self.root, split)
        self.images = images
        self.labels = labels
        self.classes = [str(i) for i in range(10)]

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, idx: int):
        x = self.images[idx].to(torch.float32).div_(255.0).unsqueeze(0)
        y = self.labels[idx]
        return x, y


train_ds = MNISTDataset(DATA_ROOT, train=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNISTDataset(DATA_ROOT, train=False)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class Classifier(nn.Module):
    def __init__(self, num_classes, dimensions, in_features, device=None, center_inputs=True):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.num_classes = num_classes
        self.dimensions = dimensions
        self.center_inputs = center_inputs
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

    def encode(self, x: Tensor) -> torchhd.BSCTensor:
        # Flatten input
        x = x.view(x.size(0), -1)
        # 输入数据中心化 (对 FPGA 友好，且有助于二值化)
        if self.center_inputs:
            x = x - 0.5
        
        # Project to high-dimensional space
        sample_hv = self.projection(x)
        # Binarize (threshold > 0)
        # 二值化处理：大于0设为True，否则为False (BSCTensor)
        return torchhd.BSCTensor(sample_hv > 0)

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

accuracies = []
for i in range(RUNS):
    torch.manual_seed(i)
    print(f"\n--- Run {i+1}/{RUNS} (Seed={i}) ---")
    
    model = Classifier(
        len(train_ds.classes),
        DIMENSIONS,
        IMG_SIZE * IMG_SIZE,
        device=device,
        center_inputs=not args.no_center,
    )
    model.fit(train_ld)
    
    print("Testing model...")
    acc = model.accuracy(test_ld)
    accuracies.append(acc)
    print(f"Run {i+1} Accuracy: {acc * 100:.3f}%")

avg_acc = statistics.mean(accuracies)
std_dev = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0

print(f"\n>>> Average Accuracy over {RUNS} runs: {avg_acc * 100:.3f}%")
print(f">>> Std Dev: {std_dev * 100:.3f}")
