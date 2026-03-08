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
from torch import Tensor
from torchhd.datasets import UCIHAR
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import index


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
        # x = x - 0.5 # UCIHAR is already centered [-1, 1]
        sample_hv = self.projection(x)
        return torchhd.BSCTensor(sample_hv > 0)

    def fit(self, data_loader, sample_weights: torch.Tensor = None) -> "Classifier":
        self.train()
        
        # 1. Pre-compute encoded samples (one pass)
        encoded_list = []
        labels_list = []
        with torch.no_grad():
            for samples, labels in data_loader:
                samples = samples.to(self.device).float()
                encoded_list.append(self.encode(samples))
                labels_list.append(labels.to(self.device))
        
        total_samples = sum(t.size(0) for t in labels_list)
        
        # Initialize class accumulators
        class_accumulators = torch.zeros((self.num_classes, self.dimensions), device=self.device)

        # Shuffle data for better training (especially if loader is not shuffled)
        # We need to shuffle encoded_list, labels_list, and sample_weights synchronously
        # It's easier to concat first, then shuffle, unless OOM is strict.
        # For strict OOM, we would shuffle indices.
        
        # Threshold set to 10000 as requested
        if total_samples > 10000:
            # --- Batch Mode (Iterative Update) ---
            # print("Using Batch Mode (OOM Protection)")
            # We can't concat all. But we can shuffle the order of batches? 
            # Or shuffle indices and pick from list? 
            # List access is slow.
            # Simple approach: Shuffle the list of batches? No, batches are small (1).
            # If batch_size=1, shuffling the list is equivalent to shuffling samples.
            
            combined = list(zip(encoded_list, labels_list))
            
            # If we have weights, we need to attach them before shuffling
            if sample_weights is not None:
                # sample_weights is a Tensor. Split it into list.
                # Assuming data_loader was sequential, sample_weights corresponds 1-to-1
                weights_list = sample_weights.split(1) # Assuming batch_size=1
                if len(weights_list) == len(combined):
                     combined_w = list(zip(encoded_list, labels_list, weights_list))
                     random.shuffle(combined_w)
                     encoded_list, labels_list, weights_list = zip(*combined_w)
                     sample_weights_shuffled = torch.cat(weights_list)
                else:
                    # Mismatch size (e.g. loader batch_size != 1 or other issue)
                    # Fallback: don't shuffle or warn.
                    # For now, assume batch_size=1 as in our script.
                    random.shuffle(combined)
                    encoded_list, labels_list = zip(*combined)
                    sample_weights_shuffled = sample_weights # Mismatch risk!
            else:
                random.shuffle(combined)
                encoded_list, labels_list = zip(*combined)
                sample_weights_shuffled = None

            # Initial Training (Batch)
            start_idx = 0
            for i, (encoded_batch, labels_batch) in enumerate(zip(encoded_list, labels_list)):
                batch_size = labels_batch.size(0)
                bipolar_batch = torch.where(
                    encoded_batch,
                    torch.tensor(1.0, device=self.device, dtype=torch.float32),
                    torch.tensor(-1.0, device=self.device, dtype=torch.float32),
                )
                
                # Apply sample weights if provided
                if sample_weights_shuffled is not None:
                    batch_weights = sample_weights_shuffled[start_idx : start_idx + batch_size].to(self.device)
                    bipolar_batch = bipolar_batch * batch_weights.unsqueeze(1)
                
                class_accumulators.index_add_(0, labels_batch, bipolar_batch)
                start_idx += batch_size

            # Retraining Loop (Batch)
            current_lr = self.lr
            prev_mistakes = total_samples + 1
            no_improve_epochs = 0
            
            for epoch in range(self.epochs):
                mistake_count = 0
                # Update centroids for the epoch
                centroids = torchhd.BSCTensor(class_accumulators >= 0)
                
                start_idx = 0
                for i, (encoded_batch, labels_batch) in enumerate(zip(encoded_list, labels_list)):
                    batch_size = labels_batch.size(0)
                    sims = torchhd.hamming_similarity(encoded_batch, centroids)
                    preds = torch.argmax(sims, dim=-1)
                    
                    # Margin Logic
                    sim_correct = sims.gather(1, labels_batch.view(-1, 1)).squeeze()
                    sims_clone = sims.clone()
                    min_val = torch.iinfo(sims.dtype).min if sims.dtype in [torch.int32, torch.int64] else -float('inf')
                    sims_clone.scatter_(1, labels_batch.view(-1, 1), min_val)
                    sim_wrong_max, wrong_max_idx = sims_clone.max(dim=1)

                    # Normalize for margin check
                    sim_correct_norm = sim_correct.float() / self.dimensions
                    sim_wrong_max_norm = sim_wrong_max.float() / self.dimensions

                    mask_wrong = preds != labels_batch
                    mask_margin = (sim_correct_norm - sim_wrong_max_norm < self.margin) & (preds == labels_batch)
                    mask_update = mask_wrong | mask_margin
                    
                    mistake_count += mask_wrong.sum().item()
                    
                    if mask_update.any():
                        push_target = torch.where(mask_wrong, preds, wrong_max_idx)
                        
                        bipolar_wrong = torch.where(
                            encoded_batch[mask_update],
                            torch.tensor(1.0, device=self.device, dtype=torch.float32),
                            torch.tensor(-1.0, device=self.device, dtype=torch.float32),
                        )
                        update_vec = bipolar_wrong * current_lr
                        
                        if sample_weights_shuffled is not None:
                            batch_weights = sample_weights_shuffled[start_idx : start_idx + batch_size].to(self.device)
                            update_vec = update_vec * batch_weights[mask_update].unsqueeze(1)
                        
                        class_accumulators.index_add_(0, labels_batch[mask_update], update_vec)
                        class_accumulators.index_add_(0, push_target[mask_update], -update_vec)
                    
                    start_idx += batch_size
            
            # Early Stopping Check
                if mistake_count < 0.001 * total_samples:
                    break
                if mistake_count >= prev_mistakes:
                    no_improve_epochs += 1
                else:
                    no_improve_epochs = 0
                prev_mistakes = mistake_count
                if no_improve_epochs >= 2:
                    break
            
                current_lr = self.lr / (1.0 + 0.1 * (epoch + 1))

        else:
            # --- Full Mode (Vectorized) ---
            encoded_samples = torch.cat(encoded_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            # Shuffle in Full Mode
            perm = torch.randperm(total_samples)
            encoded_samples = encoded_samples[perm]
            all_labels = all_labels[perm]
            if sample_weights is not None:
                weights_all = sample_weights.to(self.device)[perm]
            else:
                weights_all = None

            # Polarization Consistency
            bipolar_all = torch.where(
                encoded_samples,
                torch.tensor(1.0, device=self.device, dtype=torch.float32),
                torch.tensor(-1.0, device=self.device, dtype=torch.float32),
            )
            
            # Initial one-shot learning
            if weights_all is not None:
                bipolar_weighted = bipolar_all * weights_all.unsqueeze(1)
                class_accumulators.index_add_(0, all_labels, bipolar_weighted)
            else:
                class_accumulators.index_add_(0, all_labels, bipolar_all)
            
            # Retraining Loop
            current_lr = self.lr
            prev_mistakes = total_samples + 1
            no_improve_epochs = 0
            
            for epoch in range(self.epochs):
                centroids = torchhd.BSCTensor(class_accumulators >= 0)
                
                sims = torchhd.hamming_similarity(encoded_samples, centroids)
                preds = torch.argmax(sims, dim=-1)
                
                # Margin Logic
                sim_correct = sims.gather(1, all_labels.view(-1, 1)).squeeze()
                sims_clone = sims.clone()
                min_val = torch.iinfo(sims.dtype).min if sims.dtype in [torch.int32, torch.int64] else -float('inf')
                sims_clone.scatter_(1, all_labels.view(-1, 1), min_val)
                sim_wrong_max, wrong_max_idx = sims_clone.max(dim=1)

                # Normalize for margin check
                sim_correct_norm = sim_correct.float() / self.dimensions
                sim_wrong_max_norm = sim_wrong_max.float() / self.dimensions

                mask_wrong = preds != all_labels
                mask_margin = (sim_correct_norm - sim_wrong_max_norm < self.margin) & (preds == all_labels)
                mask_update = mask_wrong | mask_margin

                mistake_count = mask_wrong.sum().item()
                
                # Early Stopping
                if mistake_count == 0:
                    break
                # if mistake_count < 0.001 * total_samples:
                #     break
                # if mistake_count >= prev_mistakes:
                #     no_improve_epochs += 1
                # else:
                #     no_improve_epochs = 0
                # prev_mistakes = mistake_count
                # if no_improve_epochs >= 2:
                #     break
                
                # Vectorized update
                if mask_update.any():
                    push_target = torch.where(mask_wrong, preds, wrong_max_idx)

                    bipolar_wrong = bipolar_all[mask_update]
                    update_vec = bipolar_wrong * current_lr
                    
                    if weights_all is not None:
                        weights_wrong = weights_all[mask_update]
                        update_vec = update_vec * weights_wrong.unsqueeze(1)
                    
                    class_accumulators.index_add_(0, all_labels[mask_update], update_vec)
                    class_accumulators.index_add_(0, push_target[mask_update], -update_vec)
                
                current_lr = self.lr / (1.0 + 0.1 * (epoch + 1))

        self.centroids = torchhd.BSCTensor(class_accumulators >= 0)
        return self

    def forward(self, samples: Tensor) -> Tensor:
        return torchhd.hamming_similarity(self.encode(samples), self.centroids)

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)


def split_train_val(
    dataset: UCIHAR, val_ratio: float, seed: int
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
        if isinstance(name, str) and name.startswith("har_"):
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
        if isinstance(name, str) and name.startswith("har_"):
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
    epochs: int = 0,
    sample_weights: torch.Tensor = None,
    margin: float = 0.0,
) -> Classifier:
    torch.manual_seed(seed)
    feature_indices = torch.from_numpy(idx_e).long().to(device)
    model = Classifier(
        num_classes=NUM_CLASSES,
        dimensions=D_e,
        in_features=len(idx_e),
        device=device,
        feature_indices=feature_indices,
        epochs=epochs,
        margin=margin,
    )
    model.fit(train_loader, sample_weights=sample_weights)
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
    epochs: int = 0,
    margin: float = 0.0,
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
            epochs=epochs,
            sample_weights=None, # Use resampling instead of reweighting
            margin=margin,
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
    epochs: int = 0,
    margin: float = 0.0,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    data_dir = os.path.join(base_dir, "data")
    train_ds = UCIHAR(data_dir, train=True, download=True)
    test_ds = UCIHAR(data_dir, train=False, download=True)

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
        f"Starting {num_runs} runs (D_total={D_total}, E={E}, weighting={weighting}, margin={margin})",
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
            epochs=epochs,
            margin=margin,
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
                epochs=epochs,
                margin=margin,
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
                    epochs=epochs,
                    margin=margin,
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
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.0)
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
        os.path.join(parent_dir, "rf_feature_importance_results/har_feature_importance.csv")
    )

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    effective_num_runs = args.runs if args.runs is not None else args.num_runs
    
    filename = f"har_fi_D{args.D_total}_E{args.E}_{args.weighting}_rr{int(args.per_expert_ratio * 100)}"
    if args.margin > 0:
        filename += f"_margin{args.margin}"
    filename += f"_run{args.run_time}.csv"
    
    results_path = os.path.join(results_dir, filename)
    
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
        epochs=args.epochs,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
