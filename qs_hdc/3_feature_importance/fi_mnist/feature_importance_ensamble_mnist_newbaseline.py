import argparse
import os
import csv
import json
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchhd
import torchvision
from torch import Tensor
from torchvision.datasets import MNIST
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import index


INPUT_FEATURES = 28 * 28
NUM_CLASSES = 10


class Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dimensions: int,
        in_features: int,
        device: torch.device,
        feature_indices: torch.Tensor = None,
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.dimensions = dimensions
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

    def fit(self, data_loader) -> "Classifier":
        self.train()
        class_accumulators = torch.zeros(
            self.num_classes,
            self.dimensions,
            dtype=torch.int32,
            device=self.device,
        )
        with torch.no_grad():
            for samples, labels in data_loader:
                samples = samples.to(self.device).float()
                labels = labels.to(self.device)
                bipolar = torch.where(
                    self.encode(samples),
                    torch.tensor(1, device=self.device, dtype=torch.int32),
                    torch.tensor(-1, device=self.device, dtype=torch.int32),
                )
                class_accumulators.index_add_(0, labels, bipolar)
        self.centroids = torchhd.BSCTensor(class_accumulators > 0)
        return self

    def forward(self, samples: Tensor) -> Tensor:
        return torchhd.hamming_similarity(self.encode(samples), self.centroids)

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)


def split_train_val(
    dataset, val_ratio: float, seed: int
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
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


def load_feature_importance_indices(
    csv_path: str,
    n_features: int,
    common_ratio: float,
    per_expert_ratio: float,
    E: int,
) -> List[np.ndarray]:
    df = pd.read_csv(csv_path)
    if "rank_perm" not in df.columns:
        raise RuntimeError("Expected column 'rank_perm' in feature importance CSV.")
    df_sorted = df.sort_values(by=["rank_perm", "rank_avg"], ascending=[True, True])
    feature_indices = []
    for name in df_sorted["feature"].tolist():
        if isinstance(name, str) and name.startswith("px_"):
            idx = int(name.split("_")[1])
        else:
            idx = int(name)
        feature_indices.append(idx)
    feature_indices = np.array(feature_indices, dtype=np.int64)
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
    df = pd.read_csv(csv_path)
    if "rank_perm" not in df.columns:
        raise RuntimeError("Expected column 'rank_perm' in feature importance CSV.")
    df_sorted = df.sort_values(by=["rank_perm", "rank_avg"], ascending=[True, True])
    feature_indices = []
    for name in df_sorted["feature"].tolist():
        if isinstance(name, str) and name.startswith("px_"):
            idx = int(name.split("_")[1])
        else:
            idx = int(name)
        feature_indices.append(idx)
    feature_indices = np.array(feature_indices, dtype=np.int64)
    if feature_indices.shape[0] < n_features:
        raise RuntimeError("Feature importance CSV does not cover all features.")
    if feature_indices.shape[0] > n_features:
        feature_indices = feature_indices[:n_features]
    return feature_indices


def train_expert(
    idx_e: np.ndarray,
    D_e: int,
    seed: int,
    train_loader,
    device: torch.device,
) -> Classifier:
    torch.manual_seed(seed)
    feature_indices = torch.from_numpy(idx_e).long().to(device)
    model = Classifier(
        num_classes=NUM_CLASSES,
        dimensions=D_e,
        in_features=len(idx_e),
        device=device,
        feature_indices=feature_indices,
    )
    model.fit(train_loader)
    return model


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


def soft_vote_weighted(
    experts: List[Classifier],
    samples: Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    acc_sim = None
    for i, model in enumerate(experts):
        sim = model(samples) * weights[i]
        acc_sim = sim if acc_sim is None else acc_sim + sim
    return torch.argmax(acc_sim, dim=-1)


def train_experts_boosting(
    E: int,
    partitions: List[np.ndarray],
    dims: List[int],
    train_subset,
    train_eval_loader,
    device: torch.device,
    expert_seeds: List[int],
) -> Tuple[List[Classifier], List[float]]:
    n_train = len(train_subset)
    sample_weights = np.ones(n_train, dtype=np.float64) / n_train
    experts: List[Classifier] = []
    alphas: List[float] = []

    for e in range(E):
        weights_tensor = torch.from_numpy(sample_weights).float()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights_tensor, num_samples=n_train, replacement=True
        )
        train_loader_weighted = torch.utils.data.DataLoader(
            train_subset, batch_size=1, sampler=sampler
        )

        expert = train_expert(
            idx_e=partitions[e],
            D_e=dims[e],
            seed=expert_seeds[e],
            train_loader=train_loader_weighted,
            device=device,
        )

        preds = []
        labels_all = []
        with torch.no_grad():
            for samples, labels in train_eval_loader:
                samples = samples.to(device).float()
                labels = labels.to(device)
                pred = expert.predict(samples)
                preds.append(pred.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)

        incorrect = preds != labels_all
        weighted_error = float((sample_weights * incorrect).sum())

        if weighted_error <= 0.0:
            alpha = 5.0
        elif weighted_error >= 0.5:
            alpha = 1e-3
        else:
            alpha = 0.5 * float(
                np.log((1.0 - weighted_error) / max(weighted_error, 1e-8))
            )

        sample_weights *= np.exp(alpha * incorrect.astype(np.float64))
        sample_weights /= sample_weights.sum() + 1e-12

        experts.append(expert)
        alphas.append(alpha)

    return experts, alphas


def generate_seeds_for_models(num_models: int, master_seed: int) -> List[int]:
    rnd = random.Random(master_seed)
    return rnd.sample(range(10000), num_models)


def run_experiment(
    D_total: int,
    E: int,
    run_time: int,
    device: torch.device,
    num_runs: int,
    val_ratio: float,
    weighting: str,
    results_csv_path: str,
    importance_csv_path: str,
    common_ratio: float,
    per_expert_ratio: float,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    data_dir = os.path.join(base_dir, "data")
    transform = torchvision.transforms.ToTensor()
    train_ds = MNIST(data_dir, train=True, transform=transform, download=True)
    test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

    if E <= 0:
        E = 1
    base_dim = D_total // E
    dims = [base_dim for _ in range(E)]
    residue = D_total - base_dim * E
    for i in range(residue):
        dims[i] += 1

    partitions = load_feature_importance_indices(
        csv_path=importance_csv_path,
        n_features=INPUT_FEATURES,
        common_ratio=common_ratio,
        per_expert_ratio=per_expert_ratio,
        E=E,
    )
    sorted_all = parse_sorted_feature_indices(
        csv_path=importance_csv_path, n_features=INPUT_FEATURES
    )

    baseline_k_global = len(partitions[0])
    hw_bits = index.compute_hardware_bits(dims, partitions, E, NUM_CLASSES, D_total, baseline_k_global)
    runs_metrics: List[dict] = []
    seeds_list: List[List[int]] = []

    print(
        f"Starting {num_runs} runs (D_total={D_total}, E={E}, weighting={weighting})",
        flush=True,
    )

    for r in range(num_runs):
        print(f"[Run {r + 1}/{num_runs}] Training and evaluation...", flush=True)
        seed_base = run_time + r

        train_subset, val_subset = split_train_val(
            train_ds, val_ratio=val_ratio, seed=seed_base
        )

        train_ld = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
        val_ld = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False)
        test_ld = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

        all_seeds = generate_seeds_for_models(E + 1, seed_base)
        baseline_seed = all_seeds[0]
        expert_seeds = all_seeds[1:]

        torch.manual_seed(baseline_seed)
        np.random.seed(baseline_seed)
        baseline_k = len(partitions[0])
        baseline_idx = sorted_all[:baseline_k]
        baseline = Classifier(
            NUM_CLASSES,
            D_total,
            baseline_k,
            device=device,
            feature_indices=torch.from_numpy(baseline_idx).long().to(device),
        )
        baseline.fit(train_ld)
        base_acc = evaluate_accuracy(baseline, test_ld)

        if weighting == "boosting":
            train_eval_ld = torch.utils.data.DataLoader(
                train_subset, batch_size=1, shuffle=False
            )
            experts, alphas = train_experts_boosting(
                E=E,
                partitions=partitions,
                dims=dims,
                train_subset=train_subset,
                train_eval_loader=train_eval_ld,
                device=device,
                expert_seeds=expert_seeds,
            )
            w = torch.tensor(alphas, device=device, dtype=torch.float32)
        else:
            experts: List[Classifier] = []
            for e in range(E):
                model_e = train_expert(
                    idx_e=partitions[e],
                    D_e=dims[e],
                    seed=expert_seeds[e],
                    train_loader=train_ld,
                    device=device,
                )
                experts.append(model_e)

            if weighting == "val_acc":
                expert_accs_val: List[float] = []
                for e in range(E):
                    acc_val = evaluate_accuracy(experts[e], val_ld)
                    expert_accs_val.append(acc_val)
                w = torch.tensor(expert_accs_val, device=device, dtype=torch.float32)
                w = w / (w.sum() + 1e-8)
            else:
                w = torch.ones(E, device=device, dtype=torch.float32) / E

        metrics = index.evaluate_run(experts, baseline, w, test_ld, device, NUM_CLASSES)
        runs_metrics.append(metrics)
        seeds_list.append([baseline_seed] + expert_seeds)
        print(
            f"[Run {r + 1}/{num_runs}] "
            f"baseline_seed={baseline_seed}, expert_seeds={expert_seeds}, "
            f"baseline_acc={metrics['base_acc']:.4f}, ensemble_acc={metrics['ens_acc']:.4f}",
            flush=True,
        )
        index.print_run_metrics(metrics, hw_bits)

    agg = index.aggregate_runs(runs_metrics)
    index.write_csv_row(
        results_csv_path=results_csv_path,
        dims=dims,
        E=E,
        D_total=D_total,
        run_time=run_time,
        weighting=weighting,
        importance_csv_path=importance_csv_path,
        common_ratio=common_ratio,
        per_expert_ratio=per_expert_ratio,
        baseline_k=baseline_k_global,
        seeds_list=seeds_list,
        agg=agg,
        hw_bits=hw_bits,
        num_runs=num_runs,
    )
    print(f"Baseline mean accuracy: {agg['acc_baseline_mean']:.4f}, std: {agg['acc_baseline_std']:.4f}")
    print(f"Ensemble mean accuracy: {agg['acc_ensemble_mean']:.4f}, std: {agg['acc_ensemble_std']:.4f}")
    for i, acc in enumerate(agg['expert_accs_mean']):
        print(f"Expert {i} mean accuracy: {acc:.4f}")
    print(f"Saved summary to: {results_csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D_total", type=int, default=6000)
    parser.add_argument("--E", type=int, default=3)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--weighting", type=str, default="boosting", choices=["uniform", "val_acc", "boosting"]
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--common_ratio", type=float, default=0.10)
    parser.add_argument("--per_expert_ratio", type=float, default=0.30)
    parser.add_argument("--baseline_k_mode", type=str, default="first", choices=["first", "avg"])
    args = parser.parse_args()

    if args.E != 3:
        print("Warning: 当前脚本设计假定 E=3（示例中的 Expert1/2/3 分牌方式），已使用传入的 E。")

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    importance_csv_path = os.path.normpath(
        os.path.join(parent_dir, "rf_feature_importance_results/mnist_feature_importance.csv")
    )

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    effective_num_runs = args.runs if args.runs is not None else args.num_runs
    results_path = os.path.join(
        results_dir,
        f"mnist_fi_D{args.D_total}_E{args.E}_{args.weighting}_rr{int(args.per_expert_ratio * 100)}_run{args.run_time}.csv",
    )
    run_experiment(
        D_total=args.D_total,
        E=args.E,
        run_time=args.run_time,
        device=device,
        num_runs=effective_num_runs,
        val_ratio=args.val_ratio,
        weighting=args.weighting,
        results_csv_path=results_path,
        importance_csv_path=importance_csv_path,
        common_ratio=args.common_ratio,
        per_expert_ratio=args.per_expert_ratio,
    )


if __name__ == "__main__":
    main()
