import os
import csv
import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def evaluate_run(experts, baseline, weights, test_loader, device, num_classes: int):
    n_total = 0
    n_correct = 0
    expert_correct = [0 for _ in range(len(experts))]
    labels_all = []
    ens_preds = []
    base_preds = []
    expert_preds_list = [[] for _ in range(len(experts))]
    ens_margins_correct = []
    ens_margins_wrong = []
    base_margins_correct = []
    base_margins_wrong = []
    with torch.no_grad():
        for samples, labels in test_loader:
            samples = samples.to(device).float()
            labels = labels.to(device)
            sim_acc = None
            for i, model in enumerate(experts):
                sim_i = model(samples) * weights[i]
                sim_acc = sim_i if sim_acc is None else sim_acc + sim_i
            pred_ens = torch.argmax(sim_acc, dim=-1)
            top2e, _ = torch.topk(sim_acc.squeeze(0), k=2, largest=True, sorted=True)
            margin_e = float(top2e[0].item() - top2e[1].item())
            pred_base = baseline.predict(samples)
            sim_base = baseline(samples)
            top2b, _ = torch.topk(sim_base.squeeze(0), k=2, largest=True, sorted=True)
            margin_b = float(top2b[0].item() - top2b[1].item())
            n_total += labels.size(0)
            correct_ens = (pred_ens == labels)
            n_correct += torch.sum(correct_ens).item()
            labels_all.append(labels.cpu().numpy())
            ens_preds.append(pred_ens.cpu().numpy())
            base_preds.append(pred_base.cpu().numpy())
            if bool(correct_ens.all().item()):
                ens_margins_correct.append(margin_e)
            else:
                ens_margins_wrong.append(margin_e)
            correct_base = (pred_base == labels)
            if bool(correct_base.all().item()):
                base_margins_correct.append(margin_b)
            else:
                base_margins_wrong.append(margin_b)
            for e in range(len(experts)):
                pred_e = experts[e].predict(samples)
                expert_correct[e] += torch.sum(pred_e == labels).item()
                expert_preds_list[e].append(pred_e.cpu().numpy())
    labels_all = np.concatenate(labels_all, axis=0)
    ens_preds = np.concatenate(ens_preds, axis=0)
    base_preds = np.concatenate(base_preds, axis=0)
    cm_ens = confusion_matrix(labels_all, ens_preds, labels=list(range(num_classes)))
    cm_base = confusion_matrix(labels_all, base_preds, labels=list(range(num_classes)))
    expert_errors = []
    for e in range(len(experts)):
        preds_e = np.concatenate(expert_preds_list[e], axis=0)
        err_e = (preds_e != labels_all).astype(np.float32)
        expert_errors.append(err_e)
    expert_errors = np.stack(expert_errors, axis=0)
    with np.errstate(invalid="ignore"):
        corr_err = np.corrcoef(expert_errors)
    corr_err = np.nan_to_num(corr_err, nan=0.0)
    disagree_rates = []
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            preds_i = np.concatenate(expert_preds_list[i], axis=0)
            preds_j = np.concatenate(expert_preds_list[j], axis=0)
            disagree_rates.append(np.mean(preds_i != preds_j))
    mean_disagree = float(np.mean(disagree_rates)) if disagree_rates else 0.0
    all_ens_margins = ens_margins_correct + ens_margins_wrong
    all_base_margins = base_margins_correct + base_margins_wrong
    mean_ens_margin = float(np.mean(all_ens_margins)) if all_ens_margins else 0.0
    mean_base_margin = float(np.mean(all_base_margins)) if all_base_margins else 0.0
    mean_ens_margin_correct = (
        float(np.mean(ens_margins_correct)) if ens_margins_correct else 0.0
    )
    mean_ens_margin_wrong = (
        float(np.mean(ens_margins_wrong)) if ens_margins_wrong else 0.0
    )
    mean_base_margin_correct = (
        float(np.mean(base_margins_correct)) if base_margins_correct else 0.0
    )
    mean_base_margin_wrong = (
        float(np.mean(base_margins_wrong)) if base_margins_wrong else 0.0
    )
    gap_ens_margin = mean_ens_margin_correct - mean_ens_margin_wrong
    gap_base_margin = mean_base_margin_correct - mean_base_margin_wrong
    ens_acc = n_correct / n_total
    base_acc = float(np.mean(base_preds == labels_all))
    expert_accs = [x / n_total for x in expert_correct]
    return {
        "base_acc": base_acc,
        "ens_acc": ens_acc,
        "expert_accs": expert_accs,
        "cm_ens": cm_ens,
        "cm_base": cm_base,
        "corr_err": corr_err,
        "mean_ens_margin": mean_ens_margin,
        "mean_base_margin": mean_base_margin,
        "mean_disagree": mean_disagree,
        "mean_ens_margin_correct": mean_ens_margin_correct,
        "mean_ens_margin_wrong": mean_ens_margin_wrong,
        "mean_base_margin_correct": mean_base_margin_correct,
        "mean_base_margin_wrong": mean_base_margin_wrong,
        "gap_ens_margin": gap_ens_margin,
        "gap_base_margin": gap_base_margin,
    }


def compute_hardware_bits(dims, partitions, E, num_classes, D_total, baseline_k):
    proj_bits_ens = 0
    for i in range(E):
        proj_bits_ens += dims[i] * len(partitions[i])
    cent_bits_ens = E * num_classes * int(np.mean(dims))
    proj_bits_base = D_total * baseline_k
    cent_bits_base = num_classes * D_total

    # Hardware-oriented metrics
    # Memory footprint (bits)
    # Model storage: Projection matrix (usually fixed/generated on fly for HDC) + Centroids
    # Here we assume projection matrix is generated on the fly (LFSR), so we only count model weights (Centroids)
    # However, for FPGA, if we store the projection matrix, it consumes significant BRAM.
    # Let's provide both: "Model Size" (Centroids only) and "Total Memory" (if Projection stored)
    
    # Quantization assumptions: 
    # - Projection matrix: Binary {-1, 1} -> 1 bit per element (or generated, 0 bits storage)
    # - Centroids: Integer/Float. Standard HDC uses integers. Let's assume 16-bit integers for accumulation.
    q_bits = 16 
    
    model_size_bits_ens = cent_bits_ens * q_bits
    model_size_bits_base = cent_bits_base * q_bits
    
    # Latency / Operations per query
    # Operations = Projection Ops + Similarity Ops
    # Projection Ops: Input_dim * D_dim (XOR/Add)
    # Similarity Ops: D_dim * Num_Classes (Hamming/Cosine)
    
    # For Ensemble: Sum of experts
    ops_proj_ens = 0
    ops_sim_ens = 0
    for i in range(E):
        d_i = dims[i]
        k_i = len(partitions[i])
        ops_proj_ens += k_i * d_i
        ops_sim_ens += d_i * num_classes
        
    ops_proj_base = baseline_k * D_total
    ops_sim_base = D_total * num_classes
    
    total_ops_ens = ops_proj_ens + ops_sim_ens
    total_ops_base = ops_proj_base + ops_sim_base
    
    return {
        "proj_bits_ens": int(proj_bits_ens),
        "cent_bits_ens": int(cent_bits_ens),
        "proj_bits_base": int(proj_bits_base),
        "cent_bits_base": int(cent_bits_base),
        "model_size_bits_ens": int(model_size_bits_ens),
        "model_size_bits_base": int(model_size_bits_base),
        "total_ops_ens": int(total_ops_ens),
        "total_ops_base": int(total_ops_base),
        "ops_reduction": float(1.0 - total_ops_ens / total_ops_base) if total_ops_base > 0 else 0.0
    }


def print_run_metrics(metrics, hw_bits):
    print("Confusion Matrix (Ensemble):")
    print(metrics["cm_ens"])
    print("Confusion Matrix (Baseline):")
    print(metrics["cm_base"])
    print("Error Correlation Matrix (Experts):")
    print(metrics["corr_err"])
    print(
        f"Confidence Margin (ensemble) -> all={metrics['mean_ens_margin']:.6f}, "
        f"correct={metrics['mean_ens_margin_correct']:.6f}, "
        f"wrong={metrics['mean_ens_margin_wrong']:.6f}, "
        f"gap={metrics['gap_ens_margin']:.6f}"
    )
    print(
        f"Confidence Margin (baseline) -> all={metrics['mean_base_margin']:.6f}, "
        f"correct={metrics['mean_base_margin_correct']:.6f}, "
        f"wrong={metrics['mean_base_margin_wrong']:.6f}, "
        f"gap={metrics['gap_base_margin']:.6f}"
    )
    print(
        f"Expert Diversity (Disagreement Rate avg) -> {metrics['mean_disagree']:.6f}"
    )
    print(
        f"Hardware Cost (bits) -> Ensemble: Projection={hw_bits['proj_bits_ens']}, Centroid={hw_bits['cent_bits_ens']}; "
        f"Baseline: Projection={hw_bits['proj_bits_base']}, Centroid={hw_bits['cent_bits_base']}"
    )
    print(
        f"FPGA/Edge Efficiency -> Ops Reduction: {hw_bits['ops_reduction']*100:.2f}%; "
        f"Model Size (16-bit): {hw_bits['model_size_bits_ens']/8/1024:.2f} KB (Ens) vs {hw_bits['model_size_bits_base']/8/1024:.2f} KB (Base)"
    )


def aggregate_runs(runs_metrics):
    base_accs = [m["base_acc"] for m in runs_metrics]
    ens_accs = [m["ens_acc"] for m in runs_metrics]
    expert_accs_runs = [m["expert_accs"] for m in runs_metrics]
    base_acc_mean = float(np.mean(base_accs)) if base_accs else 0.0
    ens_acc_mean = float(np.mean(ens_accs)) if ens_accs else 0.0
    base_acc_std = float(np.std(base_accs)) if base_accs else 0.0
    ens_acc_std = float(np.std(ens_accs)) if ens_accs else 0.0
    expert_accs_mean = list(np.mean(np.array(expert_accs_runs), axis=0)) if expert_accs_runs else []
    mean_ens_margin_over_runs = float(np.mean([m["mean_ens_margin"] for m in runs_metrics])) if runs_metrics else 0.0
    mean_base_margin_over_runs = float(np.mean([m["mean_base_margin"] for m in runs_metrics])) if runs_metrics else 0.0
    mean_ens_margin_correct_over_runs = float(
        np.mean([m["mean_ens_margin_correct"] for m in runs_metrics])
    ) if runs_metrics else 0.0
    mean_ens_margin_wrong_over_runs = float(
        np.mean([m["mean_ens_margin_wrong"] for m in runs_metrics])
    ) if runs_metrics else 0.0
    mean_base_margin_correct_over_runs = float(
        np.mean([m["mean_base_margin_correct"] for m in runs_metrics])
    ) if runs_metrics else 0.0
    mean_base_margin_wrong_over_runs = float(
        np.mean([m["mean_base_margin_wrong"] for m in runs_metrics])
    ) if runs_metrics else 0.0
    gap_ens_margin_over_runs = mean_ens_margin_correct_over_runs - mean_ens_margin_wrong_over_runs
    gap_base_margin_over_runs = mean_base_margin_correct_over_runs - mean_base_margin_wrong_over_runs
    mean_disagree_over_runs = float(np.mean([m["mean_disagree"] for m in runs_metrics])) if runs_metrics else 0.0
    cm_ens_sum = None
    cm_base_sum = None
    corr_err_sum = None
    corr_count = 0
    for m in runs_metrics:
        cm_ens = m["cm_ens"].astype(np.int64)
        cm_base = m["cm_base"].astype(np.int64)
        corr = m["corr_err"].astype(np.float64)
        if cm_ens_sum is None:
            cm_ens_sum = cm_ens
            cm_base_sum = cm_base
            corr_err_sum = corr
        else:
            cm_ens_sum += cm_ens
            cm_base_sum += cm_base
            corr_err_sum += corr
        corr_count += 1
    cm_ens_final = cm_ens_sum.tolist() if cm_ens_sum is not None else []
    cm_base_final = cm_base_sum.tolist() if cm_base_sum is not None else []
    corr_err_avg = (corr_err_sum / max(corr_count, 1)).tolist() if corr_err_sum is not None else []
    return {
        "acc_baseline_mean": base_acc_mean,
        "acc_baseline_std": base_acc_std,
        "acc_ensemble_mean": ens_acc_mean,
        "acc_ensemble_std": ens_acc_std,
        "expert_accs_mean": expert_accs_mean,
        "mean_confidence_margin_ens": mean_ens_margin_over_runs,
        "mean_confidence_margin_base": mean_base_margin_over_runs,
        "mean_confidence_margin_ens_correct": mean_ens_margin_correct_over_runs,
        "mean_confidence_margin_ens_wrong": mean_ens_margin_wrong_over_runs,
        "mean_confidence_margin_base_correct": mean_base_margin_correct_over_runs,
        "mean_confidence_margin_base_wrong": mean_base_margin_wrong_over_runs,
        "gap_confidence_margin_ens": gap_ens_margin_over_runs,
        "gap_confidence_margin_base": gap_base_margin_over_runs,
        "mean_disagree_rate": mean_disagree_over_runs,
        "cm_ens": cm_ens_final,
        "cm_base": cm_base_final,
        "corr_err": corr_err_avg,
    }


def write_csv_row(
    results_csv_path: str,
    dims,
    E: int,
    D_total: int,
    run_time: int,
    weighting: str,
    importance_csv_path: str,
    common_ratio: float,
    per_expert_ratio: float,
    baseline_k: int,
    seeds_list,
    agg,
    hw_bits,
    num_runs: int,
):
    if results_csv_path.endswith(".csv"):
        base_path = results_csv_path[:-4]
    else:
        base_path = results_csv_path
    main_path = base_path + "_main.csv"
    extra_path = base_path + "_extra.csv"

    main_row = {
        "D_total": D_total,
        "E": E,
        "num_runs": num_runs,
        "acc_baseline_mean": agg["acc_baseline_mean"],
        "acc_baseline_std": agg["acc_baseline_std"],
        "acc_ensemble_mean": agg["acc_ensemble_mean"],
        "acc_ensemble_std": agg["acc_ensemble_std"],
    }
    for i in range(3):
        key = f"acc_e{i}_mean"
        if i < len(agg["expert_accs_mean"]):
            main_row[key] = float(agg["expert_accs_mean"][i])
        else:
            main_row[key] = 0.0

    extra_row = {
        "D_total": D_total,
        "E": E,
        "dims": json.dumps(dims),
        "run_time": run_time,
        "num_runs": num_runs,
        "weighting": weighting,
        "importance_csv": importance_csv_path,
        "common_ratio": common_ratio,
        "per_expert_ratio": per_expert_ratio,
        "baseline_k": baseline_k,
        "seeds": json.dumps(seeds_list),
        "mean_confidence_margin_ens": agg["mean_confidence_margin_ens"],
        "mean_confidence_margin_base": agg["mean_confidence_margin_base"],
        "mean_confidence_margin_ens_correct": agg["mean_confidence_margin_ens_correct"],
        "mean_confidence_margin_ens_wrong": agg["mean_confidence_margin_ens_wrong"],
        "mean_confidence_margin_base_correct": agg["mean_confidence_margin_base_correct"],
        "mean_confidence_margin_base_wrong": agg["mean_confidence_margin_base_wrong"],
        "gap_confidence_margin_ens": agg["gap_confidence_margin_ens"],
        "gap_confidence_margin_base": agg["gap_confidence_margin_base"],
        "mean_disagree_rate": agg["mean_disagree_rate"],
        "cm_ens": json.dumps(agg["cm_ens"]),
        "cm_base": json.dumps(agg["cm_base"]),
        "corr_err": json.dumps(agg["corr_err"]),
        "proj_bits_ens": hw_bits["proj_bits_ens"],
        "cent_bits_ens": hw_bits["cent_bits_ens"],
        "proj_bits_base": hw_bits["proj_bits_base"],
        "cent_bits_base": hw_bits["cent_bits_base"],
        "model_size_bits_ens": hw_bits["model_size_bits_ens"],
        "model_size_bits_base": hw_bits["model_size_bits_base"],
        "total_ops_ens": hw_bits["total_ops_ens"],
        "total_ops_base": hw_bits["total_ops_base"],
        "ops_reduction": hw_bits["ops_reduction"],
    }

    write_header_main = not os.path.exists(main_path)
    with open(main_path, "a", newline="") as f_main:
        writer_main = csv.DictWriter(f_main, fieldnames=list(main_row.keys()))
        if write_header_main:
            writer_main.writeheader()
        writer_main.writerow(main_row)

    write_header_extra = not os.path.exists(extra_path)
    with open(extra_path, "a", newline="") as f_extra:
        writer_extra = csv.DictWriter(f_extra, fieldnames=list(extra_row.keys()))
        if write_header_extra:
            writer_extra.writeheader()
        writer_extra.writerow(extra_row)
