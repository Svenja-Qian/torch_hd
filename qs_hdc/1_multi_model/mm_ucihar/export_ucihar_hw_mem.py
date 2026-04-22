import argparse
import os

import torch
import torchhd
from torchhd.datasets import UCIHAR
from tqdm import tqdm


def quantize_to_int16(x: torch.Tensor, scale: float) -> torch.Tensor:
    xq = torch.round(x * scale)
    xq = torch.clamp(xq, -32768, 32767)
    return xq.to(torch.int16)


def projection_weight_bits(in_features: int, dimensions: int, seed: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(seed)
    proj = torchhd.embeddings.Projection(in_features, dimensions)
    with torch.no_grad():
        w = proj.weight.data
        w = torch.sign(w)
        w[w == 0] = 1
        w = w.to(torch.int8)
    return (w > 0).to(device)


def export_weights_memb(weight_bits: torch.Tensor, out_path: str) -> None:
    dimensions, in_features = weight_bits.shape
    with open(out_path, "w") as f:
        for d in range(dimensions):
            bits = weight_bits[d]
            line = "".join("1" if bits[k].item() else "0" for k in range(in_features - 1, -1, -1))
            f.write(line + "\n")


def bits_to_hex_msb_first(bits: torch.Tensor) -> str:
    d = int(bits.numel())
    pad = (-d) % 4
    if pad:
        bits = torch.cat([torch.zeros(pad, dtype=torch.bool), bits.to(torch.bool)], dim=0)
    else:
        bits = bits.to(torch.bool)
    out = []
    for i in range(0, bits.numel(), 4):
        nib = bits[i : i + 4]
        v = (int(nib[0]) << 3) | (int(nib[1]) << 2) | (int(nib[2]) << 1) | int(nib[3])
        out.append(format(v, "x"))
    return "".join(out)


def export_centroids_memh(centroids_bits: torch.Tensor, out_path: str) -> None:
    num_classes, dimensions = centroids_bits.shape
    with open(out_path, "w") as f:
        for c in range(num_classes):
            bits = centroids_bits[c].to(torch.bool)
            bits_msb_first = torch.flip(bits, dims=[0])
            f.write(bits_to_hex_msb_first(bits_msb_first) + "\n")


def infer_one(
    xq: torch.Tensor,
    w_pm1: torch.Tensor,
    centroids_bits: torch.Tensor,
) -> int:
    acc = torch.sum(w_pm1 * xq.to(torch.int32).unsqueeze(0), dim=1)
    hv = acc > 0
    xnor = ~(hv.unsqueeze(0) ^ centroids_bits)
    scores = xnor.to(torch.int32).sum(dim=1)
    return int(torch.argmax(scores).item())


def export_testcases_memh(
    ds,
    weight_bits: torch.Tensor,
    centroids_bits: torch.Tensor,
    out_path: str,
    quant_scale: float,
    num_cases: int,
    offset: int,
) -> None:
    w_pm1 = torch.where(weight_bits, torch.tensor(1, dtype=torch.int32), torch.tensor(-1, dtype=torch.int32))
    w_pm1 = w_pm1.to(torch.int32)
    centroids_bits = centroids_bits.to(torch.bool)

    in_features = ds[0][0].numel()
    with open(out_path, "w") as f:
        for i in range(num_cases):
            x, _ = ds[offset + i]
            xq = quantize_to_int16(x.float(), quant_scale)
            pred = infer_one(xq, w_pm1, centroids_bits)
            f.write(f"{pred & 0xFFFF:04x}\n")
            if xq.numel() != in_features:
                raise RuntimeError("Unexpected feature length")
            for k in range(in_features):
                f.write(f"{int(xq[k].item()) & 0xFFFF:04x}\n")


def build_centroids(
    train_ld,
    weight_bits: torch.Tensor,
    dimensions: int,
    num_classes: int,
    quant_scale: float,
    device: torch.device,
    max_samples: int | None,
) -> torch.Tensor:
    w_pm1 = torch.where(weight_bits, torch.tensor(1, dtype=torch.int32), torch.tensor(-1, dtype=torch.int32))
    w_pm1 = w_pm1.to(device)

    class_acc = torch.zeros(num_classes, dimensions, dtype=torch.int32, device=device)

    n = 0
    for samples, labels in tqdm(train_ld, desc="Build centroids"):
        samples = samples.to(device).float()
        labels = labels.to(device)
        xq = quantize_to_int16(samples, quant_scale).to(torch.int32)

        acc = torch.empty((xq.size(0), dimensions), dtype=torch.int32, device=device)
        for b in range(xq.size(0)):
            acc[b] = torch.sum(w_pm1 * xq[b].unsqueeze(0), dim=1)

        hv = acc > 0
        bipolar = torch.where(hv, torch.tensor(1, device=device, dtype=torch.int32), torch.tensor(-1, device=device, dtype=torch.int32))
        class_acc.index_add_(0, labels, bipolar)

        n += xq.size(0)
        if max_samples is not None and n >= max_samples:
            break

    return (class_acc > 0).to(torch.bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.abspath("."), "data"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.abspath("."), "hw_mem_ucihar"))
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quant-scale", type=float, default=32767.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--export-testcases", type=int, default=0)
    parser.add_argument("--testcase-offset", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = UCIHAR(args.data_dir, train=True, download=True)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_ds = UCIHAR(args.data_dir, train=False, download=True)

    in_features = train_ds[0][0].numel()
    num_classes = len(train_ds.classes)

    w_bits = projection_weight_bits(in_features, args.dimensions, args.seed, device)

    weights_path = os.path.join(args.out_dir, f"weights_D{args.dimensions}_seed{args.seed}.memb")
    export_weights_memb(w_bits, weights_path)

    max_samples = None if args.max_train_samples == 0 else int(args.max_train_samples)
    centroids = build_centroids(
        train_ld,
        w_bits,
        args.dimensions,
        num_classes,
        args.quant_scale,
        device,
        max_samples,
    )

    centroids_path = os.path.join(args.out_dir, f"centroids_D{args.dimensions}_seed{args.seed}.memh")
    export_centroids_memh(centroids, centroids_path)

    testcases_path = ""
    if args.export_testcases > 0:
        testcases_path = os.path.join(
            args.out_dir,
            f"testcases_D{args.dimensions}_seed{args.seed}_n{args.export_testcases}_off{args.testcase_offset}.memh",
        )
        export_testcases_memh(
            test_ds,
            w_bits,
            centroids,
            testcases_path,
            args.quant_scale,
            args.export_testcases,
            args.testcase_offset,
        )

    print(weights_path)
    print(centroids_path)
    if testcases_path:
        print(testcases_path)


if __name__ == "__main__":
    main()
