"""Extract frozen RGB features for one Hoi! recording from its frame manifests."""

import argparse
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch
from torch.nn import functional as F


@torch.inference_mode()
def dino_features(paths, model, processor, device, batch_size):
    cls, grids, means = [], [], []
    for offset in range(0, len(paths), batch_size):
        images = []
        for path in paths[offset:offset + batch_size]:
            with Image.open(path) as frame:
                images.append(frame.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC))
        inputs = processor(
            images=images, return_tensors="pt", do_resize=False, do_center_crop=False
        ).pixel_values.to(device, dtype=next(model.parameters()).dtype)
        hidden = model(pixel_values=inputs).last_hidden_state
        patches = hidden[:, -784:].reshape(-1, 28, 28, 1024).float()
        grids.append(F.avg_pool2d(patches.permute(0, 3, 1, 2), 4).permute(0, 2, 3, 1).half().cpu())
        means.append(patches.mean(dim=(1, 2)).half().cpu())
        cls.append(hidden[:, 0].half().cpu())
    return tuple(torch.cat(parts).numpy() for parts in (cls, grids, means))


def da3_windows(length):
    size = min(16, length)
    starts = list(range(0, length - size + 1, 8))
    if starts[-1] != length - size:
        starts.append(length - size)
    frames = np.arange(length)
    starts_array = np.asarray(starts)
    contains = (starts_array <= frames[:, None]) & (frames[:, None] < starts_array + size)
    distance = np.abs(frames[:, None] - starts_array - size // 2)
    return size, starts, np.where(contains, distance, length).argmin(axis=1)


@torch.inference_mode()
def da3_features(paths, model, device):
    pooled = torch.empty((len(paths), 7, 7, 2048), dtype=torch.bfloat16)
    size, starts, owner = da3_windows(len(paths))
    for index, start in enumerate(starts):
        local = np.flatnonzero(owner == index) - start
        if not len(local):
            continue
        rgb = [str(path) for path in paths[start:start + size]]
        images, _, _ = model._preprocess_inputs(
            rgb, extrinsics=None, intrinsics=None, process_res=448,
            process_res_method="upper_bound_resize"
        )
        images, _, _ = model._prepare_model_inputs(images, None, None)
        enabled = device.type == "cuda"
        dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            levels, _ = model.model.backbone(
                images, cam_token=None, export_feat_layers=[], ref_view_strategy="middle"
            )
        height, width = images.shape[-2:]
        tokens = levels[-1][0][0, local].reshape(len(local), height // 14, width // 14, 2048)
        grid = F.adaptive_avg_pool2d(tokens.permute(0, 3, 1, 2).float(), (7, 7))
        pooled[local + start] = grid.permute(0, 2, 3, 1).to(device="cpu", dtype=pooled.dtype)
    return pooled


def extract_recording(args, dino, processor, da3, device):
    with (args.label_root / "manifest.jsonl").open(encoding="utf-8") as stream:
        clips = [json.loads(line) for line in stream if line.strip()]
    destinations = [args.output / name for name in ("compact", "spatial", "da3")]
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in destinations[:2]:
        (directory / args.recording_uid).mkdir(parents=True, exist_ok=True)
    (destinations[2] / "samples").mkdir(parents=True, exist_ok=True)
    for clip in clips:
        sample = clip["sample_id"]
        uid = f"{args.recording_uid}__{sample}"
        compact, spatial = [parent / args.recording_uid / f"{sample}.npz" for parent in destinations[:2]]
        depth = destinations[2] / "samples" / f"{uid}.safetensors"
        if any(path.exists() for path in (compact, spatial, depth)):
            raise FileExistsError(f"Feature output already exists for {uid}")
        with (args.label_root / clip["paths"]["frames"]).open(newline="", encoding="utf-8") as stream:
            frames = list(csv.DictReader(stream))
        timestamps = np.asarray([int(frame["timestamp"]) for frame in frames], dtype=np.int64)
        if not len(timestamps) or (np.diff(timestamps) <= 0).any():
            raise ValueError(f"Empty or unordered timestamp sequence for {uid}")
        paths = [args.dataset_root / frame["rgb_path"] for frame in frames]
        cls, dino_grid, dino_mean = dino_features(paths, dino, processor, device, args.batch_size)
        da3_grid = da3_features(paths, da3, device)
        valid = np.ones(len(timestamps), dtype=np.uint8)
        identity = {
            "recording_uid": args.recording_uid, "sample_id": sample,
            "frame_timestamps_ns": timestamps, "visual_valid_mask": valid,
        }
        with TemporaryDirectory(dir=args.output) as staging:
            stage = Path(staging)
            tokens = np.zeros((len(timestamps), 5, 1024), dtype=np.float16)
            tokens[:, 0] = cls
            np.savez(stage / compact.name, **identity, tokens=tokens,
                     visual_token_names=np.array(["cls", "unused1", "unused2", "unused3", "unused4"]))
            np.savez(stage / "spatial.npz", **identity, dino_grid_7x7=dino_grid,
                     dino_patch_mean=dino_mean)
            save_file({
                "da3_intermediate_grid_7x7": da3_grid.contiguous(),
                "frame_timestamps_ns": torch.from_numpy(timestamps),
                "da3_token_valid_mask": torch.from_numpy(valid),
            }, stage / depth.name, metadata={"sample_uid": uid})
            for source, target in ((stage / compact.name, compact),
                                   (stage / "spatial.npz", spatial), (stage / depth.name, depth)):
                os.link(source, target)
        print(uid, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--recording-uid", required=True)
    parser.add_argument("--dino", required=True, help="Local DINOv3 ViT-L/16 checkpoint")
    parser.add_argument("--da3", required=True, help="Local DA3-LARGE-1.1 checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    from transformers import AutoImageProcessor, AutoModel
    from depth_anything_3.api import DepthAnything3

    device = torch.device(args.device)
    torch.set_num_threads(1)
    processor = AutoImageProcessor.from_pretrained(args.dino, local_files_only=True)
    dino = AutoModel.from_pretrained(args.dino, local_files_only=True).to(device).eval()
    da3 = DepthAnything3.from_pretrained(args.da3).to(device).eval()
    extract_recording(args, dino, processor, da3, device)


if __name__ == "__main__":
    main()
