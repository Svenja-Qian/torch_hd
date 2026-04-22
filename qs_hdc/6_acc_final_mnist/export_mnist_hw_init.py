import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
QS_HDC_DIR = CURRENT_DIR.parent
FI_MNIST_DIR = QS_HDC_DIR / "3_feature_importance" / "fi_mnist"
if str(FI_MNIST_DIR) not in sys.path:
    sys.path.insert(0, str(FI_MNIST_DIR))

from feature_importance_ensamble_mnist_retrain_v3_margin_boosting import (  # noqa: E402
    INPUT_FEATURES,
    LocalMNIST,
    MNIST,
    NUM_CLASSES,
    generate_seeds_for_models,
    parse_sorted_feature_indices,
    split_train_val,
    train_expert,
    train_experts_boosting,
)


def sample_to_uint8_flat(sample: torch.Tensor) -> torch.Tensor:
    sample_flat = sample.view(-1).float()
    return torch.round(sample_flat * 255.0).to(torch.int16)


def build_hardware_partitions(
    csv_path: str,
    n_features: int,
    common_ratio: float,
    per_expert_ratio: float,
    num_experts: int,
) -> dict[str, Any]:
    feature_indices = parse_sorted_feature_indices(csv_path, n_features)
    common_count = max(1, int(n_features * common_ratio))
    features_per_expert = max(1, int(n_features * per_expert_ratio))
    unique_per_expert = max(0, features_per_expert - common_count)

    common_indices = feature_indices[:common_count].astype(np.int64)
    remaining = feature_indices[common_count:]

    total_unique_needed = unique_per_expert * num_experts
    if total_unique_needed > remaining.shape[0]:
        total_unique_needed = remaining.shape[0]
        unique_per_expert = total_unique_needed // num_experts
        total_unique_needed = unique_per_expert * num_experts

    unique_pool = remaining[:total_unique_needed]
    expert_unique: list[np.ndarray] = []
    partitions: list[np.ndarray] = []
    for expert_id in range(num_experts):
        unique_idx = unique_pool[expert_id:total_unique_needed:num_experts].astype(np.int64)
        expert_unique.append(unique_idx)
        partitions.append(
            np.concatenate([common_indices, unique_idx], axis=0).astype(np.int64)
        )

    return {
        "common_indices": common_indices,
        "expert_unique": expert_unique,
        "partitions": partitions,
        "common_count": int(common_indices.shape[0]),
        "private_count": int(unique_per_expert),
        "features_per_expert": int(common_indices.shape[0] + unique_per_expert),
    }


def pack_bits_to_word(bits: torch.Tensor) -> int:
    word = 0
    for bit_idx, bit in enumerate(bits.tolist()):
        if bit:
            word |= 1 << bit_idx
    return word


def chunk_bool_vector(bits: torch.Tensor, chunk_width: int) -> list[int]:
    if bits.ndim != 1:
        raise ValueError("Expected a 1D bit vector.")
    if bits.numel() % chunk_width != 0:
        raise ValueError("Bit vector size must be divisible by chunk width.")
    words: list[int] = []
    for base in range(0, bits.numel(), chunk_width):
        words.append(pack_bits_to_word(bits[base : base + chunk_width].to(torch.bool)))
    return words


def export_projection_mem(model, out_path: Path, hv_dim: int, chunk_width: int) -> None:
    weight = model.projection.weight.detach().cpu()
    if weight.shape[0] != hv_dim:
        raise RuntimeError(
            f"Projection dimension mismatch: model={weight.shape[0]}, hv_dim={hv_dim}"
        )

    num_rows = weight.shape[1]
    with out_path.open("w", encoding="ascii") as f:
        for row_idx in range(num_rows):
            row_bits = (weight[:, row_idx] > 0).to(torch.bool)
            for word in chunk_bool_vector(row_bits, chunk_width):
                f.write(f"{word:08x}\n")


def export_centroid_mem(model, out_path: Path, hv_dim: int, chunk_width: int) -> None:
    if model.centroids is None:
        raise RuntimeError("Model centroids are not available.")
    centroids = model.centroids.to(torch.bool).detach().cpu()
    if centroids.shape[1] != hv_dim:
        raise RuntimeError(
            f"Centroid dimension mismatch: model={centroids.shape[1]}, hv_dim={hv_dim}"
        )

    with out_path.open("w", encoding="ascii") as f:
        for class_idx in range(centroids.shape[0]):
            for word in chunk_bool_vector(centroids[class_idx], chunk_width):
                f.write(f"{word:08x}\n")


def quantize_boost_weights(
    alphas: list[float], weight_width: int, frac_bits: int
) -> tuple[list[int], list[bool]]:
    scale = 1 << frac_bits
    max_value = (1 << weight_width) - 1
    q_weights: list[int] = []
    clipped: list[bool] = []
    for alpha in alphas:
        raw = int(round(alpha * scale))
        q = min(max(raw, 0), max_value)
        q_weights.append(q)
        clipped.append(q != raw)
    return q_weights, clipped


def export_boost_mem(
    q_weights: list[int], out_path: Path, packed_out_path: Path, weight_width: int
) -> None:
    hex_digits = max(1, (weight_width + 3) // 4)
    with out_path.open("w", encoding="ascii") as f:
        for weight in q_weights:
            f.write(f"{weight:0{hex_digits}x}\n")

    packed = 0
    for expert_id, weight in enumerate(q_weights):
        packed |= int(weight) << (expert_id * weight_width)
    packed_digits = max(1, (len(q_weights) * weight_width + 3) // 4)
    with packed_out_path.open("w", encoding="ascii") as f:
        f.write(f"{packed:0{packed_digits}x}\n")


def export_feature_map(metadata: dict[str, Any], out_dir: Path) -> None:
    common_indices = metadata["common_indices"]
    expert_unique = metadata["expert_unique"]

    shared_path = out_dir / "shared_features.txt"
    with shared_path.open("w", encoding="ascii") as f:
        for row_idx, raw_idx in enumerate(common_indices):
            f.write(f"{row_idx},{int(raw_idx)},shared\n")

    for expert_id, unique_indices in enumerate(expert_unique):
        path = out_dir / f"feature_map_e{expert_id}.txt"
        with path.open("w", encoding="ascii") as f:
            for row_idx, raw_idx in enumerate(common_indices):
                f.write(f"{row_idx},{int(raw_idx)},shared\n")
            row_base = len(common_indices)
            for offset, raw_idx in enumerate(unique_indices):
                f.write(f"{row_base + offset},{int(raw_idx)},private\n")


def compute_expected_class_for_sample(experts, sample: torch.Tensor, q_weights: list[int]) -> int:
    weighted_dist = None
    with torch.no_grad():
        for expert_id, expert in enumerate(experts):
            hv = expert.encode(sample.unsqueeze(0)).to(torch.bool).cpu().squeeze(0)
            centroids = expert.centroids.to(torch.bool).detach().cpu()
            dist = (hv.unsqueeze(0) ^ centroids).to(torch.int32).sum(dim=1)
            score = dist * int(q_weights[expert_id])
            weighted_dist = score if weighted_dist is None else (weighted_dist + score)
    return int(torch.argmin(weighted_dist).item())


def export_testcases(
    test_ds,
    experts,
    q_weights: list[int],
    partition_info: dict[str, Any],
    out_dir: Path,
    num_cases: int,
    offset: int,
) -> dict[str, Any]:
    common_indices = partition_info["common_indices"]
    expert_unique = partition_info["expert_unique"]
    total_stream_words = 1 + len(common_indices) + sum(len(x) for x in expert_unique)

    raw_path = out_dir / "testcases_raw_pixels.mem"
    stream_path = out_dir / "testcases_hw_stream.mem"
    meta_path = out_dir / "testcases.json"

    cases_meta: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="ascii") as raw_f, stream_path.open("w", encoding="ascii") as stream_f:
        for case_id in range(num_cases):
            sample, label = test_ds[offset + case_id]
            sample_u8 = sample_to_uint8_flat(sample)
            expected_class = compute_expected_class_for_sample(experts, sample, q_weights)

            raw_f.write(f"{expected_class & 0xffff:04x}\n")
            for value in sample_u8.tolist():
                raw_f.write(f"{int(value) & 0xffff:04x}\n")

            stream_f.write(f"{expected_class & 0xffff_ffff:08x}\n")

            stream_entries: list[tuple[int, int, str]] = []
            for row_idx, raw_idx in enumerate(common_indices.tolist()):
                value = int(sample_u8[raw_idx].item())
                stream_entries.append((row_idx, value, "shared"))
            for expert_id, unique_indices in enumerate(expert_unique):
                for offset_idx, raw_idx in enumerate(unique_indices.tolist()):
                    row_idx = len(common_indices) + offset_idx
                    value = int(sample_u8[raw_idx].item())
                    stream_entries.append((row_idx, value, f"private_e{expert_id}"))

            for row_idx, value, _phase in stream_entries:
                packed = ((int(row_idx) & 0xffff) << 16) | (int(value) & 0xffff)
                stream_f.write(f"{packed:08x}\n")

            cases_meta.append(
                {
                    "case_id": case_id,
                    "dataset_index": offset + case_id,
                    "label": int(label),
                    "expected_class": expected_class,
                    "stream_words": total_stream_words,
                }
            )

    with meta_path.open("w", encoding="ascii") as f:
        json.dump(
            {
                "num_cases": num_cases,
                "offset": offset,
                "raw_pixels_file": raw_path.name,
                "hw_stream_file": stream_path.name,
                "raw_case_stride_words": 1 + INPUT_FEATURES,
                "hw_stream_case_stride_words": total_stream_words,
                "cases": cases_meta,
            },
            f,
            indent=2,
        )

    return {
        "raw_pixels_file": raw_path.name,
        "hw_stream_file": stream_path.name,
        "metadata_file": meta_path.name,
        "num_cases": num_cases,
        "offset": offset,
        "raw_case_stride_words": 1 + INPUT_FEATURES,
        "hw_stream_case_stride_words": total_stream_words,
    }


def choose_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(CURRENT_DIR / "generated" / "mnist_hw_init"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-time", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--E", type=int, default=3)
    parser.add_argument("--expert-dim", type=int, default=2048)
    parser.add_argument("--common-ratio", type=float, default=0.10)
    parser.add_argument("--per-expert-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--weighting", type=str, default="boosting", choices=["boosting", "uniform", "val_acc"])
    parser.add_argument("--chunk-width", type=int, default=32)
    parser.add_argument("--boost-weight-width", type=int, default=8)
    parser.add_argument("--boost-frac-bits", type=int, default=6)
    parser.add_argument("--export-testcases", type=int, default=10)
    parser.add_argument("--testcase-offset", type=int, default=0)
    args = parser.parse_args()

    device = choose_device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    importance_csv_path = QS_HDC_DIR / "3_feature_importance" / "rf_feature_importance_results" / "mnist_feature_importance.csv"
    data_dir = FI_MNIST_DIR / "data"

    partition_info = build_hardware_partitions(
        csv_path=str(importance_csv_path),
        n_features=INPUT_FEATURES,
        common_ratio=args.common_ratio,
        per_expert_ratio=args.per_expert_ratio,
        num_experts=args.E,
    )
    partitions = partition_info["partitions"]
    dims = [args.expert_dim for _ in range(args.E)]

    mnist_cls = MNIST if MNIST is not None else LocalMNIST
    train_ds = mnist_cls(str(data_dir), train=True, download=True, transform=None)
    test_ds = mnist_cls(str(data_dir), train=False, download=True, transform=None)
    train_subset, val_subset = split_train_val(train_ds, val_ratio=args.val_ratio, seed=args.run_time)
    train_ld = torch.utils.data.DataLoader(train_subset, batch_size=196, shuffle=True)
    train_eval_ld = torch.utils.data.DataLoader(train_subset, batch_size=196, shuffle=False)
    val_ld = torch.utils.data.DataLoader(val_subset, batch_size=196, shuffle=False)
    _ = val_ld
    _ = test_ds

    all_seeds = generate_seeds_for_models(args.E + 1, args.run_time)
    expert_seeds = all_seeds[1:]

    if args.weighting == "boosting":
        experts, alphas = train_experts_boosting(
            E=args.E,
            partitions=partitions,
            dims=dims,
            train_subset=train_subset,
            train_eval_loader=train_eval_ld,
            device=device,
            expert_seeds=expert_seeds,
            epochs=args.epochs,
            margin=args.margin,
        )
    else:
        experts = []
        for expert_id in range(args.E):
            expert = train_expert(
                idx_e=partitions[expert_id],
                D_e=dims[expert_id],
                seed=expert_seeds[expert_id],
                train_loader=train_ld,
                device=device,
                epochs=args.epochs,
                margin=args.margin,
            )
            experts.append(expert)
        if args.weighting == "val_acc":
            alphas = []
            for expert in experts:
                expert.eval()
                n_correct = 0
                n_total = 0
                with torch.no_grad():
                    for samples, labels in val_ld:
                        samples = samples.to(device).float()
                        labels = labels.to(device)
                        n_correct += torch.sum(expert.predict(samples) == labels).item()
                        n_total += labels.size(0)
                alphas.append((n_correct / max(n_total, 1)) if n_total > 0 else 0.0)
        else:
            alphas = [1.0 for _ in range(args.E)]

    q_weights, clipped = quantize_boost_weights(
        alphas, args.boost_weight_width, args.boost_frac_bits
    )

    for expert_id, expert in enumerate(experts):
        export_projection_mem(
            expert,
            out_dir / f"proj{expert_id}.mem",
            hv_dim=args.expert_dim,
            chunk_width=args.chunk_width,
        )
        export_centroid_mem(
            expert,
            out_dir / f"centroid{expert_id}.mem",
            hv_dim=args.expert_dim,
            chunk_width=args.chunk_width,
        )

    export_boost_mem(
        q_weights,
        out_dir / "boost_weights.mem",
        out_dir / "boost_weights_packed.mem",
        args.boost_weight_width,
    )
    export_feature_map(partition_info, out_dir)
    testcase_info = None
    if args.export_testcases > 0:
        testcase_info = export_testcases(
            test_ds=test_ds,
            experts=experts,
            q_weights=q_weights,
            partition_info=partition_info,
            out_dir=out_dir,
            num_cases=args.export_testcases,
            offset=args.testcase_offset,
        )

    manifest = {
        "dataset": "mnist",
        "preprocess": "raw_totensor_no_centering",
        "weighting": args.weighting,
        "num_experts": args.E,
        "num_classes": NUM_CLASSES,
        "input_features": INPUT_FEATURES,
        "expert_dim": args.expert_dim,
        "chunk_width": args.chunk_width,
        "num_chunks": args.expert_dim // args.chunk_width,
        "common_ratio": args.common_ratio,
        "per_expert_ratio": args.per_expert_ratio,
        "shared_count": partition_info["common_count"],
        "private_count_per_expert": partition_info["private_count"],
        "features_per_expert": partition_info["features_per_expert"],
        "run_time": args.run_time,
        "expert_seeds": expert_seeds,
        "importance_csv_path": str(importance_csv_path),
        "common_indices": [int(x) for x in partition_info["common_indices"]],
        "expert_unique_indices": [
            [int(x) for x in unique_idx] for unique_idx in partition_info["expert_unique"]
        ],
        "expert_row_to_raw_feature": [
            [int(x) for x in part] for part in partition_info["partitions"]
        ],
        "boost_alpha_float": [float(alpha) for alpha in alphas],
        "boost_alpha_q": q_weights,
        "boost_alpha_clipped": clipped,
        "testcases": testcase_info,
        "files": {
            "projection": [f"proj{expert_id}.mem" for expert_id in range(args.E)],
            "centroid": [f"centroid{expert_id}.mem" for expert_id in range(args.E)],
            "boost_per_expert": "boost_weights.mem",
            "boost_packed": "boost_weights_packed.mem",
            "shared_features": "shared_features.txt",
            "feature_maps": [f"feature_map_e{expert_id}.txt" for expert_id in range(args.E)],
        },
    }
    if testcase_info is not None:
        manifest["files"]["testcase_raw_pixels"] = testcase_info["raw_pixels_file"]
        manifest["files"]["testcase_hw_stream"] = testcase_info["hw_stream_file"]
        manifest["files"]["testcase_metadata"] = testcase_info["metadata_file"]

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="ascii") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported hardware init files to: {out_dir}")
    print(f"Manifest: {manifest_path}")
    for expert_id in range(args.E):
        print(out_dir / f"proj{expert_id}.mem")
        print(out_dir / f"centroid{expert_id}.mem")
    print(out_dir / "boost_weights.mem")
    print(out_dir / "boost_weights_packed.mem")
    if testcase_info is not None:
        print(out_dir / testcase_info["raw_pixels_file"])
        print(out_dir / testcase_info["hw_stream_file"])
        print(out_dir / testcase_info["metadata_file"])


if __name__ == "__main__":
    main()
