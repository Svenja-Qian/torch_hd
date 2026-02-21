import argparse
import os
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchhd
from torch import Tensor
from torchhd.datasets import ISOLET

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import index


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
        if isinstance(name, str) and name.startswith("iso_"):
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
    unique_per_expert = max(1, int(n_features * per_expert_ratio))
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
        if isinstance(name, str) and name.startswith("iso_"):
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


def topk_margin(scores: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    top2, _ = torch.topk(scores, k=2, dim=-1, largest=True, sorted=True)
    top1 = top2[..., 0]
    top2_val = top2[..., 1]
    margin = top1 - top2_val
    return top1, top2_val, margin


def confidence_weighted_vote(
    experts: List[Classifier],
    samples: Tensor,
    base_weights: torch.Tensor,
    mode: str,
    tau: float,
    clamp_max: float,
    margin_bins: torch.Tensor,
    expert_margin_th: float,
):
    device = samples.device
    E = len(experts)
    all_margins = []
    all_confs = []
    acc_sim = None
    for i, model in enumerate(experts):
        scores_i = model(samples)
        top1_i, _, margin_i = topk_margin(scores_i)
        top1_i = top1_i.to(torch.float32)
        margin_i = margin_i.to(torch.float32)
        if mode == "top1":
            base_val = top1_i
        elif mode == "hybrid":
            base_val = 0.5 * (top1_i + margin_i)
        else:
            base_val = margin_i
        t0, t1, t2 = margin_bins[0], margin_bins[1], margin_bins[2]
        conf_i = torch.zeros_like(base_val, device=device)
        conf_i = conf_i + (base_val >= t0).float()
        conf_i = conf_i + (base_val >= t1).float()
        conf_i = conf_i + (base_val >= t2).float()
        if expert_margin_th > 0.0:
            conf_i = conf_i * (margin_i >= expert_margin_th).float()
        if tau != 1.0:
            conf_i = conf_i * float(tau)
        if clamp_max > 0.0:
            conf_i = torch.clamp(conf_i, max=float(clamp_max))
        all_margins.append(margin_i)
        all_confs.append(conf_i)
        w_eff = base_weights[i] * conf_i
        w_eff = w_eff.unsqueeze(-1)
        weighted_scores = scores_i * w_eff
        acc_sim = weighted_scores if acc_sim is None else acc_sim + weighted_scores
    if acc_sim is None:
        raise RuntimeError("No experts provided for confidence_weighted_vote.")
    conf_matrix = torch.stack(all_confs, dim=0)
    active_matrix = conf_matrix > 0.0
    active_counts_per_sample = active_matrix.sum(dim=0).to(torch.float32)
    avg_active_experts = float(active_counts_per_sample.mean().item())
    preds = torch.argmax(acc_sim, dim=-1)
    debug = {
        "scores": acc_sim,
        "per_expert_mean_margin": [float(m.mean().item()) for m in all_margins],
        "per_expert_mean_conf": [float(c.mean().item()) for c in all_confs],
        "active_experts_per_sample": active_counts_per_sample.detach().cpu(),
        "avg_active_experts": avg_active_experts,
    }
    return preds, debug


def predict_with_filter(
    experts: List[Classifier],
    baseline: Classifier,
    samples: Tensor,
    base_weights: torch.Tensor,
    cfg: dict,
):
    device = samples.device
    margin_bins = torch.tensor(
        cfg["margin_bins"], dtype=torch.float32, device=device
    )
    preds_ens, debug = confidence_weighted_vote(
        experts=experts,
        samples=samples,
        base_weights=base_weights,
        mode=cfg["conf_vote_mode"],
        tau=float(cfg["conf_tau"]),
        clamp_max=float(cfg["conf_clamp_max"]),
        margin_bins=margin_bins,
        expert_margin_th=float(cfg["expert_margin_th"]),
    )
    scores = debug["scores"]
    top1, _, ens_margin = topk_margin(scores)
    ens_margin_th = float(cfg["ens_margin_th"])
    is_confident = ens_margin >= ens_margin_th
    use_baseline_fallback = bool(cfg["use_baseline_fallback"])
    if use_baseline_fallback:
        pred_base = baseline.predict(samples)
        pred_final = torch.where(is_confident, preds_ens, pred_base)
        fallback_used = ~is_confident
    else:
        pred_final = preds_ens
        fallback_used = ~is_confident
    info = {
        "is_confident": is_confident.detach().cpu(),
        "fallback_used": fallback_used.detach().cpu(),
        "ens_margin": ens_margin.detach().cpu(),
        "ens_top1": top1.detach().cpu(),
        "active_experts_per_sample": debug[
            "active_experts_per_sample"
        ],
        "avg_active_experts": debug["avg_active_experts"],
    }
    return pred_final, info


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
    conf_cfg: dict,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    data_dir = os.path.join(base_dir, "data")
    train_ds = ISOLET(data_dir, train=True, download=True)
    test_ds = ISOLET(data_dir, train=False, download=True)

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

        metrics = index.evaluate_run(
            experts, baseline, w, test_ld, device, NUM_CLASSES
        )

        if conf_cfg.get("use_conf_weighted_vote", False):
            margin_bins_np = conf_cfg.get("margin_bins", [2.0, 5.0, 10.0])
            acc_drop_tolerance = 0.002
            all_val_margins = []
            all_val_labels = []
            all_val_ens_preds = []
            all_val_base_preds = []
            with torch.no_grad():
                for samples_val, labels_val in val_ld:
                    samples_val = samples_val.to(device).float()
                    labels_val = labels_val.to(device)
                    margin_bins_tensor = torch.tensor(
                        margin_bins_np, dtype=torch.float32, device=device
                    )
                    preds_val_ens, debug_val = confidence_weighted_vote(
                        experts=experts,
                        samples=samples_val,
                        base_weights=w,
                        mode=conf_cfg["conf_vote_mode"],
                        tau=float(conf_cfg["conf_tau"]),
                        clamp_max=float(conf_cfg["conf_clamp_max"]),
                        margin_bins=margin_bins_tensor,
                        expert_margin_th=float(conf_cfg["expert_margin_th"]),
                    )
                    scores_val = debug_val["scores"]
                    top1_val, _, margin_val = topk_margin(scores_val)
                    pred_val_base = baseline.predict(samples_val)
                    all_val_margins.append(margin_val.detach().cpu().numpy())
                    all_val_labels.append(labels_val.detach().cpu().numpy())
                    all_val_ens_preds.append(preds_val_ens.detach().cpu().numpy())
                    all_val_base_preds.append(pred_val_base.detach().cpu().numpy())
            if all_val_labels:
                val_margins_np = np.concatenate(all_val_margins, axis=0)
                val_labels_np = np.concatenate(all_val_labels, axis=0)
                val_ens_preds_np = np.concatenate(all_val_ens_preds, axis=0)
                val_base_preds_np = np.concatenate(all_val_base_preds, axis=0)
                ens_acc_val = float(
                    np.mean(val_ens_preds_np == val_labels_np)
                )
                if conf_cfg.get("ens_margin_th", -1.0) >= 0.0:
                    best_th = float(conf_cfg["ens_margin_th"])
                    best_fallback_rate = float(0.0)
                    best_acc_filtered = ens_acc_val
                else:
                    percentiles = [50, 60, 70, 80, 90, 95]
                    cand_ths = sorted(
                        set(
                            float(np.percentile(val_margins_np, p))
                            for p in percentiles
                        )
                    )
                    if not cand_ths:
                        cand_ths = [0.0]
                    best_th = cand_ths[0]
                    best_fallback_rate = -1.0
                    best_acc_filtered = ens_acc_val
                    for th in cand_ths:
                        mask_conf = val_margins_np >= th
                        if conf_cfg.get("use_baseline_fallback", True):
                            final_preds = np.where(
                                mask_conf, val_ens_preds_np, val_base_preds_np
                            )
                        else:
                            final_preds = val_ens_preds_np
                        acc_th = float(
                            np.mean(final_preds == val_labels_np)
                        )
                        fallback_rate_th = float(
                            np.mean(~mask_conf)
                        )
                        if acc_th >= ens_acc_val - acc_drop_tolerance:
                            if fallback_rate_th > best_fallback_rate:
                                best_fallback_rate = fallback_rate_th
                                best_th = float(th)
                                best_acc_filtered = acc_th
                conf_cfg_run = dict(conf_cfg)
                conf_cfg_run["ens_margin_th"] = best_th
                total_filtered_correct = 0
                total_filtered = 0
                all_fb_flags = []
                all_conf_flags = []
                all_active_counts = []
                with torch.no_grad():
                    for samples_te, labels_te in test_ld:
                        samples_te = samples_te.to(device).float()
                        labels_te = labels_te.to(device)
                        preds_filtered, info = predict_with_filter(
                            experts=experts,
                            baseline=baseline,
                            samples=samples_te,
                            base_weights=w,
                            cfg=conf_cfg_run,
                        )
                        total_filtered_correct += torch.sum(
                            preds_filtered == labels_te
                        ).item()
                        total_filtered += labels_te.size(0)
                        all_fb_flags.append(
                            info["fallback_used"].numpy().astype(np.bool_)
                        )
                        all_conf_flags.append(
                            info["is_confident"].numpy().astype(np.bool_)
                        )
                        all_active_counts.append(
                            info["active_experts_per_sample"].numpy().astype(
                                np.float32
                            )
                        )
                if total_filtered > 0:
                    filtered_acc = total_filtered_correct / total_filtered
                else:
                    filtered_acc = 0.0
                if all_fb_flags:
                    fb_vec = np.concatenate(all_fb_flags, axis=0)
                    conf_vec = np.concatenate(all_conf_flags, axis=0)
                    active_vec = np.concatenate(
                        all_active_counts, axis=0
                    )
                    fallback_rate_test = float(np.mean(fb_vec))
                    confident_rate_test = float(np.mean(conf_vec))
                    avg_active_experts_test = float(
                        np.mean(active_vec)
                    )
                else:
                    fallback_rate_test = 0.0
                    confident_rate_test = 0.0
                    avg_active_experts_test = 0.0
                metrics["filtered_ens_acc"] = filtered_acc
                metrics["val_ens_acc_conf_vote"] = ens_acc_val
                metrics["val_filtered_acc_best"] = best_acc_filtered
                metrics["val_fallback_rate_best"] = best_fallback_rate
                metrics["ens_margin_th_used"] = best_th
                metrics["fallback_rate_test"] = fallback_rate_test
                metrics["confident_rate_test"] = confident_rate_test
                metrics["avg_active_experts_test"] = avg_active_experts_test
                print(
                    f"Confidence-filtered ensemble -> acc={filtered_acc:.4f}, "
                    f"val_acc_no_filter={ens_acc_val:.4f}, "
                    f"val_acc_filtered_best={best_acc_filtered:.4f}, "
                    f"val_fallback_rate_best={best_fallback_rate:.4f}, "
                    f"ens_margin_th_used={best_th:.4f}"
                )
                print(
                    f"Test fallback_rate={fallback_rate_test:.4f}, "
                    f"confident_rate={confident_rate_test:.4f}, "
                    f"avg_active_experts={avg_active_experts_test:.4f}"
                )
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
    parser.add_argument(
        "--importance_csv",
        type=str,
        default="../rf_feature_importance_results/isolet_feature_importance.csv",
    )
    parser.add_argument("--common_ratio", type=float, default=0.10)
    parser.add_argument("--per_expert_ratio", type=float, default=0.20)
    parser.add_argument("--baseline_k_mode", type=str, default="first", choices=["first", "avg"])
    parser.add_argument(
        "--use_conf_weighted_vote", type=int, default=1
    )
    parser.add_argument("--ens_margin_th", type=float, default=-1.0)
    parser.add_argument("--expert_margin_th", type=float, default=0.0)
    parser.add_argument(
        "--use_baseline_fallback", type=int, default=1
    )
    parser.add_argument(
        "--margin_bins",
        type=float,
        nargs=3,
        default=[2.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--conf_vote_mode",
        type=str,
        default="margin",
        choices=["margin", "top1", "hybrid"],
    )
    parser.add_argument("--conf_tau", type=float, default=1.0)
    parser.add_argument("--conf_clamp_max", type=float, default=3.0)
    args = parser.parse_args()

    if args.E != 3:
        print("Warning: 当前脚本设计假定 E=3（示例中的 Expert1/2/3 分牌方式），已使用传入的 E。")

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(args.importance_csv):
        importance_csv_path = args.importance_csv
    else:
        importance_csv_path = os.path.normpath(os.path.join(base_dir, args.importance_csv))

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    effective_num_runs = args.runs if args.runs is not None else args.num_runs
    conf_cfg = {
        "use_conf_weighted_vote": bool(args.use_conf_weighted_vote),
        "ens_margin_th": args.ens_margin_th,
        "expert_margin_th": args.expert_margin_th,
        "use_baseline_fallback": bool(args.use_baseline_fallback),
        "margin_bins": args.margin_bins,
        "conf_vote_mode": args.conf_vote_mode,
        "conf_tau": args.conf_tau,
        "conf_clamp_max": args.conf_clamp_max,
    }
    results_path = os.path.join(
        results_dir,
        f"isolet_fi_cw_D{args.D_total}_E{args.E}_{args.weighting}_rr{int(args.per_expert_ratio * 100)}_run{args.run_time}.csv",
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
        conf_cfg=conf_cfg,
    )


if __name__ == "__main__":
    main()
