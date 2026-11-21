"""Train, evaluate, and export EgoPhys predictions."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ClipDataset, collate, read_json, read_split
from .model import EgoPhys, TARGETS, VARIANTS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def new_output(path):
    path.mkdir(parents=True, exist_ok=False)
    return path


def train_statistics(dataset):
    values = np.stack([dataset.targets(uid) for uid in dataset.sample_ids])
    scales = [
        max(floor, float(np.quantile(positive, 0.75))) if positive.size else floor
        for values_for_target, floor in zip(values.T, (1.0, 0.05))
        for positive in (values_for_target[values_for_target > 0],)
    ]
    priors = np.maximum(np.median(values, axis=0), 1e-3).tolist()
    return scales, priors


def envelope(contact, valid):
    contact = contact & valid
    first_seen = contact.cumsum(1) > 0
    last_ahead = contact.flip(1).cumsum(1).flip(1) > 0
    return first_seen & last_ahead & valid


def masked_loss(loss, mask):
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def training_loss(outputs, batch, scales):
    predicted = torch.stack([outputs[key] for key in TARGETS], dim=1)
    regression = ((predicted - batch["target"]).abs() / scales).sum(1).mean()
    if "envelope_logits" not in outputs:
        return regression
    valid = outputs["frame_valid"]
    dense_mean = F.smooth_l1_loss(
        outputs["mean_force_n"], batch["mean_force_target"], reduction="none"
    )
    loaded = F.binary_cross_entropy_with_logits(
        outputs["envelope_logits"], envelope(batch["contact"], valid).float(),
        reduction="none"
    )
    return regression + 0.06 * (
        masked_loss(dense_mean, valid & batch["mean_force_valid"])
        + masked_loss(loaded, valid)
    )


def move(batch, device):
    return {
        key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)
    }


def autocast(device, precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def loader(dataset, args, shuffle=False):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate,
        generator=torch.Generator().manual_seed(getattr(args, "seed", 1))
    )


def metrics(target, prediction):
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid targets")
    errors = np.abs(prediction - target)
    return {
        key: {
            "mae": float(error.mean()),
            "median_ae": float(np.median(error)),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "mre": float((error / np.maximum(np.abs(truth), 1e-8)).mean())
        }
        for key, error, truth in zip(TARGETS, errors.T, target.T)
    }


@torch.inference_mode()
def infer(model, batches, device, precision):
    model.eval()
    values, targets, identities = [], [], []
    for raw in batches:
        batch = move(raw, device)
        with autocast(device, precision):
            outputs = model(batch)
        value = torch.stack([outputs[key] for key in TARGETS], dim=1)
        if not torch.isfinite(value).all() or (value < 0).any():
            raise FloatingPointError("Expected finite, non-negative predictions")
        values.append(value.cpu().numpy().astype(np.float64))
        identities.extend(zip(raw["uid"], raw["sequence"]))
        if "target" in raw:
            targets.append(raw["target"].numpy().astype(np.float64))
    predictions = np.concatenate(values, axis=0)
    return identities, predictions, metrics(np.concatenate(targets), predictions) if targets else None


def source_version():
    root = Path(__file__).parent
    fingerprint = sha256()
    for path in sorted(root.glob("*.py")):
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    try:
        checkout = root.parent
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=checkout,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        if Path(top).resolve() != checkout.resolve():
            return {"source_sha256": fingerprint.hexdigest(), "git_commit": None}
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"source_sha256": fingerprint.hexdigest(), "git_commit": commit}


def train(args, device):
    split = read_split(args.split)
    train_data = ClipDataset(args, split["train"])
    val_data = ClipDataset(args, split["val"])
    scales, priors = train_statistics(train_data)
    output = new_output(args.output)
    config = {
        "variant": args.variant, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": 0.0002, "weight_decay": 0.03,
        "min_lr": 1e-6, "gradient_clip": 5.0, "precision": args.precision,
        "torch": str(torch.__version__), **source_version(),
        "split_sha256": digest(args.split), "manifest_sha256": digest(args.manifest),
        "scales": scales, "priors": priors
    }
    write_json(output / "config.json", config)
    train_batches = loader(train_data, args, shuffle=True)
    val_batches = loader(val_data, args)
    model = EgoPhys(args.variant, priors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=config["min_lr"]
    )
    scale_tensor = torch.tensor(scales, device=device)
    best = float("inf")
    with (output / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for raw in train_batches:
                batch = move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, args.precision):
                    outputs = model(batch)
                loss = training_loss(outputs, batch, scale_tensor)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            schedule.step()
            _, _, validation = infer(model, val_batches, device, args.precision)
            score = sum(validation[key]["mae"] / scale for key, scale in zip(TARGETS, scales))
            summary = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                       "val_score": score, "validation": validation}
            if score < best:
                best = score
                temporary = output / "best.tmp"
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                            "val_score": score}, temporary)
                temporary.replace(output / "best.pt")
                write_json(output / "validation.json", summary)
            print(json.dumps(summary, allow_nan=False), file=history, flush=True)
            print(json.dumps(summary, allow_nan=False), flush=True)


def preflight(args):
    split = read_split(args.split)
    all_ids = [uid for ids in split.values() for uid in ids]
    dataset = ClipDataset(args, all_ids)
    if {uid for recording, entries in dataset.labels.items() for uid in
            (f"{recording}__{sample}" for sample in entries)} != set(all_ids):
        raise ValueError("Label inventory differs from split inventory")
    if {uid for recording, entries in dataset.geometry.items() for uid in
            (f"{recording}__{sample}" for sample in entries)} != set(all_ids):
        raise ValueError("Geometry inventory differs from split inventory")
    targets, durations = [], []
    missing_visual, missing_geometry = 0, 0
    for index in range(len(dataset)):
        sample = dataset.load(index, verify_rgb=True)
        targets.append(sample["target"].numpy())
        durations.append(len(sample["visual_valid"]))
        missing_visual += int((~sample["visual_valid"]).sum())
        missing_geometry += int((~sample["geometry_valid"]).sum())
    stats = np.stack(targets)
    report = {
        "counts": {key: len(ids) for key, ids in split.items()}, "split_sha256": digest(args.split),
        "clip_frames": [min(durations), max(durations)], "checked_rgb_frames": args.rgb_root is not None,
        "masked_rgb_frames": missing_visual, "masked_da3_frames": missing_geometry,
        "targets": {
            key: dict(zip(("minimum", "p50", "p95", "maximum"),
                          np.quantile(values, (0, .5, .95, 1)).tolist()))
            for key, values in zip(TARGETS, stats.T)
        },
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        write_json(new_output(args.output) / "preflight.json", report)


def export_submission(output, identities, values, expected):
    if [uid for uid, _ in identities] != list(expected):
        raise ValueError("Output clip IDs and requested IDs differ")
    if values.shape != (len(expected), len(TARGETS)) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid prediction shape or values")
    sequences = [name for _, name in identities]
    if len(sequences) != len(set(sequences)):
        raise ValueError("Sequence-directory names must be unique for submission export")
    submission = {name: dict(zip(TARGETS, map(float, row))) for name, row in zip(sequences, values)}
    payload = output / "predictions.json"
    write_json(payload, submission)
    with ZipFile(output / "submission.zip", "w", compression=ZIP_DEFLATED) as archive:
        archive.write(payload, arcname="predictions.json")


def evaluate_or_predict(args, device):
    labeled = args.command == "evaluate"
    if args.split:
        ids = read_split(args.split)[args.partition if labeled else "test"]
    else:
        ids = read_json(args.ids)
        if not isinstance(ids, list) or not ids:
            raise ValueError("--ids requires a nonempty JSON array of clip IDs")
    dataset = ClipDataset(args, ids, labeled=labeled)
    weights = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    cfg = weights["config"]
    model = EgoPhys(cfg["variant"], cfg["priors"]).to(device)
    model.load_state_dict(weights["model"], strict=True)
    identities, values, scores = infer(model, loader(dataset, args), device, args.precision)
    output = new_output(args.output)
    if labeled:
        write_json(output / "metrics.json", {
            "partition": args.partition, "samples": len(ids), "metrics": scores,
            "checkpoint_sha256": digest(args.checkpoint), "split_sha256": digest(args.split),
        })
        print(json.dumps(scores, indent=2), flush=True)
    else:
        export_submission(output, identities, values, ids)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "train", "evaluate", "predict"):
        part = commands.add_parser(command)
        part.add_argument("--manifest", type=Path, required=command != "predict")
        part.add_argument("--compact-dir", type=Path, required=True)
        part.add_argument("--spatial-dir", type=Path, required=True)
        part.add_argument("--da3-dir", type=Path, action="append", required=True)
        part.add_argument("--output", type=Path, required=command != "preflight")
        source = part.add_mutually_exclusive_group(required=True)
        source.add_argument("--split", type=Path)
        if command == "predict":
            source.add_argument("--ids", type=Path)
        if command == "preflight":
            part.add_argument("--rgb-root", type=Path)
        else:
            part.add_argument("--batch-size", type=int, default=2)
            part.add_argument("--device", default="cuda:0")
            part.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
        if command == "train":
            part.add_argument("--epochs", type=int, default=60)
            part.add_argument("--seed", type=int, default=1)
            part.add_argument("--variant", choices=VARIANTS, default="full")
            part.add_argument("--cache-memory", action="store_true")
        if command in ("evaluate", "predict"):
            part.add_argument("--checkpoint", type=Path, required=True)
        if command == "evaluate":
            part.add_argument("--partition", choices=("val", "test"), default="test")
    return parser.parse_args()


def main():
    args = arguments()
    torch.set_num_threads(1)
    if args.command == "preflight":
        preflight(args)
        return
    if args.batch_size < 1 or getattr(args, "epochs", 1) < 1:
        raise ValueError("Batch size and epoch count must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 unsupported")
    if args.command == "train":
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        train(args, device)
    else:
        evaluate_or_predict(args, device)


if __name__ == "__main__":
    main()
