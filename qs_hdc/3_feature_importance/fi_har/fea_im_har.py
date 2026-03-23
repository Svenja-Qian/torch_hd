import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchhd
from torch import Tensor
from torchhd.datasets import UCIHAR


INPUT_FEATURES = 561
NUM_CLASSES = 6


class Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dimensions: int,
        in_features: int,
        device: torch.device,
        feature_indices: torch.Tensor = None,
        epochs: int = 0,
        lr: float = 1.0,
        margin: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.dimensions = dimensions
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
        self.projection = torchhd.embeddings.Projection(in_features, dimensions)
        with torch.no_grad():
            self.projection.weight.data = self.projection.weight.data.sign()
            self.projection.weight.data[self.projection.weight.data == 0] = 1
        self.projection.to(self.device)
        self.centroids = None
        self.feature_indices = feature_indices

    def encode(self, x: Tensor) -> torchhd.BSCTensor:
        x = x.to(self.device).float()
        x = x.view(x.size(0), -1)
        if self.feature_indices is not None:
            x = x[:, self.feature_indices]
        x = x - 0.5
        sample_hv = self.projection(x)
        return torchhd.BSCTensor(sample_hv > 0)

    def fit(self, data_loader, sample_weights: torch.Tensor = None) -> "Classifier":
        self.train()

        encoded_list = []
        labels_list = []
        with torch.no_grad():
            for samples, labels in data_loader:
                samples = samples.to(self.device).float()
                encoded_list.append(self.encode(samples))
                labels_list.append(labels.to(self.device))

        total_samples = sum(t.size(0) for t in labels_list)
        class_accumulators = torch.zeros((self.num_classes, self.dimensions), device=self.device)

        encoded_samples = torch.cat(encoded_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        perm = torch.randperm(total_samples, device=all_labels.device)
        encoded_samples = encoded_samples[perm]
        all_labels = all_labels[perm]

        if sample_weights is not None:
            weights_all = sample_weights.to(self.device)[perm]
        else:
            weights_all = None

        bipolar_all = torch.where(
            encoded_samples,
            torch.tensor(1.0, device=self.device, dtype=torch.float32),
            torch.tensor(-1.0, device=self.device, dtype=torch.float32),
        )

        if weights_all is not None:
            bipolar_weighted = bipolar_all * weights_all.unsqueeze(1)
            class_accumulators.index_add_(0, all_labels, bipolar_weighted)
        else:
            class_accumulators.index_add_(0, all_labels, bipolar_all)

        current_lr = self.lr

        for epoch in range(self.epochs):
            centroids = torchhd.BSCTensor(class_accumulators >= 0)
            sims = torchhd.hamming_similarity(encoded_samples, centroids)
            preds = torch.argmax(sims, dim=-1)

            sim_correct = sims.gather(1, all_labels.view(-1, 1)).squeeze()
            sims_clone = sims.clone()
            min_val = torch.iinfo(sims.dtype).min if sims.dtype in [torch.int32, torch.int64] else -float("inf")
            sims_clone.scatter_(1, all_labels.view(-1, 1), min_val)
            sim_wrong_max, wrong_max_idx = sims_clone.max(dim=1)

            sim_correct_norm = sim_correct.float() / self.dimensions
            sim_wrong_max_norm = sim_wrong_max.float() / self.dimensions

            mask_wrong = preds != all_labels
            mask_margin = (sim_correct_norm - sim_wrong_max_norm < self.margin) & (preds == all_labels)
            mask_update = mask_wrong | mask_margin

            mistake_count = mask_wrong.sum().item()
            if mistake_count < 0.001 * total_samples:
                break

            if mask_update.any():
                push_target = torch.where(mask_wrong, preds, wrong_max_idx)
                bipolar_wrong = bipolar_all[mask_update]
                update_vec = bipolar_wrong * current_lr
                if weights_all is not None:
                    update_vec = update_vec * weights_all[mask_update].unsqueeze(1)
                class_accumulators.index_add_(0, all_labels[mask_update], update_vec)
                class_accumulators.index_add_(0, push_target[mask_update], -update_vec)

            current_lr = self.lr / (1.0 + 0.1 * (epoch + 1))

        self.centroids = torchhd.BSCTensor(class_accumulators >= 0)
        return self

    def forward(self, samples: Tensor) -> Tensor:
        return torchhd.hamming_similarity(self.encode(samples), self.centroids)

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)


def split_train_val(dataset: UCIHAR, val_ratio: float, seed: int):
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    return train_subset, val_subset


def _parse_feature_id(name: object) -> int:
    if isinstance(name, (int, np.integer)):
        return int(name)
    s = str(name)
    if "_" in s:
        tail = s.split("_")[-1]
        if tail.isdigit():
            return int(tail)
    return int(s)


def _read_sorted_feature_indices(csv_path: str) -> List[int]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "rank_perm" not in reader.fieldnames:
            raise RuntimeError("Expected column 'rank_perm' in feature importance CSV.")
        for r in reader:
            idx = _parse_feature_id(r.get("feature"))
            rank_perm = float(r.get("rank_perm"))
            rank_avg_raw = r.get("rank_avg")
            rank_avg = float(rank_avg_raw) if rank_avg_raw is not None and rank_avg_raw != "" else rank_perm
            rows.append((rank_perm, rank_avg, idx))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [idx for _, __, idx in rows]


def load_feature_importance_indices(
    csv_path: str,
    n_features: int,
    common_ratio: float,
    per_expert_ratio: float,
    E: int,
) -> List[np.ndarray]:
    feature_indices = np.array(_read_sorted_feature_indices(csv_path), dtype=np.int64)
    if feature_indices.shape[0] < n_features:
        raise RuntimeError("Feature importance CSV does not cover all features.")
    if feature_indices.shape[0] > n_features:
        feature_indices = feature_indices[:n_features]

    common_count = max(1, int(n_features * common_ratio))
    features_per_expert = max(1, int(n_features * per_expert_ratio))
    unique_per_expert = max(0, features_per_expert - common_count)

    common_indices = feature_indices[:common_count]
    remaining = feature_indices[common_count:]

    total_unique_needed = unique_per_expert * E
    if total_unique_needed > remaining.shape[0]:
        total_unique_needed = remaining.shape[0]
        unique_per_expert = total_unique_needed // E
        total_unique_needed = unique_per_expert * E

    unique_pool = remaining[:total_unique_needed]
    expert_unique: List[List[int]] = [[] for _ in range(E)]
    for pos, idx in enumerate(unique_pool):
        expert_id = pos % E
        expert_unique[expert_id].append(int(idx))

    partitions: List[np.ndarray] = []
    for e in range(E):
        combined = np.unique(
            np.concatenate(
                [
                    common_indices.astype(np.int64),
                    np.array(expert_unique[e], dtype=np.int64),
                ]
            )
        ).astype(np.int64)
        partitions.append(combined)
    return partitions


def parse_sorted_feature_indices(csv_path: str, n_features: int) -> np.ndarray:
    feature_indices = np.array(_read_sorted_feature_indices(csv_path), dtype=np.int64)
    if feature_indices.shape[0] < n_features:
        raise RuntimeError("Feature importance CSV does not cover all features.")
    if feature_indices.shape[0] > n_features:
        feature_indices = feature_indices[:n_features]
    return feature_indices


def evaluate_accuracy(model: Classifier, data_loader) -> float:
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for samples, labels in data_loader:
            samples = samples.to(model.device).float()
            labels = labels.to(model.device)
            n_correct += torch.sum(model.predict(samples) == labels).item()
            n_total += labels.size(0)
    return n_correct / n_total


def generate_seeds_for_models(num_models: int, master_seed: int) -> List[int]:
    rnd = random.Random(master_seed)
    return rnd.sample(range(10000), num_models)


def run_baseline(
    D_total: int,
    E: int,
    run_time: int,
    device: torch.device,
    num_runs: int,
    val_ratio: float,
    importance_csv_path: str,
    common_ratio: float,
    per_expert_ratio: float,
    epochs: int = 0,
    margin: float = 0.0,
    results_csv_path: Optional[str] = None,
):
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"

    train_ds = UCIHAR(str(data_dir), train=True, download=True)
    test_ds = UCIHAR(str(data_dir), train=False, download=True)

    if E <= 0:
        E = 1

    partitions = load_feature_importance_indices(
        csv_path=importance_csv_path,
        n_features=INPUT_FEATURES,
        common_ratio=common_ratio,
        per_expert_ratio=per_expert_ratio,
        E=E,
    )
    sorted_all = parse_sorted_feature_indices(csv_path=importance_csv_path, n_features=INPUT_FEATURES)

    baseline_k_global = len(partitions[0])
    accs: List[float] = []
    seeds_list: List[int] = []

    for r in range(num_runs):
        seed_base = run_time + r
        train_subset, _ = split_train_val(train_ds, val_ratio=val_ratio, seed=seed_base)
        train_ld = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
        test_ld = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

        baseline_seed = generate_seeds_for_models(1, seed_base)[0]
        torch.manual_seed(baseline_seed)
        np.random.seed(baseline_seed)

        baseline_idx = sorted_all[:baseline_k_global]
        baseline = Classifier(
            num_classes=NUM_CLASSES,
            dimensions=D_total,
            in_features=baseline_k_global,
            device=device,
            feature_indices=torch.from_numpy(baseline_idx).long().to(device),
            epochs=epochs,
            margin=margin,
        )
        baseline.fit(train_ld)
        base_acc = evaluate_accuracy(baseline, test_ld)
        accs.append(base_acc)
        seeds_list.append(baseline_seed)
        print(f"[Run {r + 1}/{num_runs}] baseline_seed={baseline_seed}, baseline_acc={base_acc:.4f}", flush=True)

    mean_acc = float(np.mean(accs)) if accs else 0.0
    std_acc = float(np.std(accs)) if accs else 0.0
    print(f"Baseline mean accuracy: {mean_acc:.4f}, std: {std_acc:.4f}", flush=True)

    if results_csv_path:
        out_path = Path(results_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(
                [
                    "D_total",
                    "E",
                    "run_time",
                    "num_runs",
                    "val_ratio",
                    "common_ratio",
                    "per_expert_ratio",
                    "baseline_k",
                    "epochs",
                    "margin",
                    "acc_mean",
                    "acc_std",
                    "seeds",
                    "importance_csv_path",
                ]
            )
            writer.writerow(
                [
                    D_total,
                    E,
                    run_time,
                    num_runs,
                    val_ratio,
                    common_ratio,
                    per_expert_ratio,
                    baseline_k_global,
                    epochs,
                    margin,
                    mean_acc,
                    std_acc,
                    json.dumps(seeds_list),
                    importance_csv_path,
                ]
            )
        print(f"Saved baseline summary to: {str(out_path)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D_total", type=int, default=6000)
    parser.add_argument("--E", type=int, default=3)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--common_ratio", type=float, default=0.10)
    parser.add_argument("--per_expert_ratio", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--results_csv_path", type=str, default="")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    results_dir = Path(__file__).resolve().parent
    base_dir = results_dir.parent
    parent_dir = base_dir.parent
    importance_csv_path = os.path.normpath(
        os.path.join(str(parent_dir), "rf_feature_importance_results/har_feature_importance.csv")
    )

    if args.results_csv_path:
        results_csv_path = args.results_csv_path
    else:
        filename = f"har_baseline_D{args.D_total}_E{args.E}_rr{int(args.per_expert_ratio * 100)}"
        if args.margin > 0:
            filename += f"_margin{args.margin}"
        filename += f"_run{args.run_time}.csv"
        results_csv_path = str(results_dir / filename)

    run_baseline(
        D_total=args.D_total,
        E=args.E,
        run_time=args.run_time,
        device=device,
        num_runs=args.num_runs,
        val_ratio=args.val_ratio,
        importance_csv_path=importance_csv_path,
        common_ratio=args.common_ratio,
        per_expert_ratio=args.per_expert_ratio,
        epochs=args.epochs,
        margin=args.margin,
        results_csv_path=results_csv_path,
    )


if __name__ == "__main__":
    main()
