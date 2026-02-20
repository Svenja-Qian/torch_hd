import argparse
import os
import csv
import json
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchhd
from torch import Tensor
from torchhd.datasets import ISOLET


INPUT_FEATURES = 617
NUM_CLASSES = 26


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
    dataset: ISOLET, val_ratio: float, seed: int
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


def make_partitions_general(
    E: int, n_features: int, overlap_ratio: float
) -> List[np.ndarray]:
    base_size = n_features // E
    remainder = n_features % E
    partitions: List[np.ndarray] = []
    start = 0
    for e in range(E):
        size = base_size + (1 if e < remainder else 0)
        end = start + size
        partitions.append(np.arange(start, end, dtype=np.int64))
        start = end
    if overlap_ratio <= 0.0:
        return partitions
    for e in range(1, E):
        k = max(1, int(round(len(partitions[e - 1]) * overlap_ratio)))
        overlap = partitions[e - 1][-k:]
        partitions[e] = np.unique(np.concatenate([overlap, partitions[e]])).astype(
            np.int64
        )
    return partitions


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

        incorrect = (preds != labels_all)
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
    overlap_ratio: float,
    device: torch.device,
    num_runs: int,
    val_ratio: float,
    weighting: str,
    results_csv_path: str,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    train_ds = ISOLET("../data", train=True, download=True)
    test_ds = ISOLET("../data", train=False, download=True)

    if E <= 0:
        E = 1
    base_dim = D_total // E
    dims = [base_dim for _ in range(E)]
    residue = D_total - base_dim * E
    for i in range(residue):
        dims[i] += 1

    partitions = make_partitions_general(
        E=E, n_features=INPUT_FEATURES, overlap_ratio=overlap_ratio
    )

    baseline_accs: List[float] = []
    ensemble_accs: List[float] = []
    expert_accs_runs: List[List[float]] = []

    for r in range(num_runs):
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
        baseline = Classifier(NUM_CLASSES, D_total, INPUT_FEATURES, device=device)
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

        n_correct = 0
        n_total = 0
        expert_accs_test = [0 for _ in range(E)]
        with torch.no_grad():
            for samples, labels in test_ld:
                samples = samples.to(device).float()
                labels = labels.to(device)
                pred_ens = soft_vote_weighted(experts, samples, w)
                n_correct += torch.sum(pred_ens == labels).item()
                n_total += labels.size(0)
                for e in range(E):
                    pred_e = experts[e].predict(samples)
                    expert_accs_test[e] += torch.sum(pred_e == labels).item()

        ens_acc = n_correct / n_total
        expert_accs = [x / n_total for x in expert_accs_test]

        baseline_accs.append(base_acc)
        ensemble_accs.append(ens_acc)
        expert_accs_runs.append(expert_accs)

    base_acc_mean = float(np.mean(baseline_accs))
    ens_acc_mean = float(np.mean(ensemble_accs))
    expert_accs_mean = list(np.mean(np.array(expert_accs_runs), axis=0))

    row = {
        "D_total": D_total,
        "E": E,
        "dims": json.dumps(dims),
        "run_time": run_time,
        "overlap_ratio": overlap_ratio,
        "num_runs": num_runs,
        "weighting": weighting,
        "acc_baseline_mean": base_acc_mean,
        "acc_ensemble_mean": ens_acc_mean,
    }
    for i, acc in enumerate(expert_accs_mean):
        row[f"acc_e{i}_mean"] = float(acc)

    write_header = not os.path.exists(results_csv_path)
    with open(results_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Baseline mean accuracy: {base_acc_mean:.4f}")
    print(f"Ensemble mean accuracy: {ens_acc_mean:.4f}")
    for i, acc in enumerate(expert_accs_mean):
        print(f"Expert {i} mean accuracy: {acc:.4f}")
    print(f"Saved summary to: {results_csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D_total", type=int, default=6000)
    parser.add_argument("--E", type=int, default=12)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--overlap_ratio", type=float, default=0.05)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--weighting", type=str, default="boosting", choices=["uniform", "val_acc", "boosting"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(
        results_dir,
        f"isolet_HRF_D{args.D_total}_E{args.E}_ov{int(args.overlap_ratio*100)}_{args.weighting}_run{args.run_time}.csv",
    )
    run_experiment(
        D_total=args.D_total,
        E=args.E,
        run_time=args.run_time,
        overlap_ratio=args.overlap_ratio,
        device=device,
        num_runs=args.num_runs,
        val_ratio=args.val_ratio,
        weighting=args.weighting,
        results_csv_path=results_path,
    )


if __name__ == "__main__":
    main()
