"""Read the fixed split and frame-aligned Hoi! feature caches."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .model import TARGETS

PARTITIONS = ("train", "val", "test")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_split(path: Path) -> dict[str, list[str]]:
    raw = read_json(path)
    split = {name: raw[name] for name in PARTITIONS}
    ids = [uid for name in PARTITIONS for uid in split[name]]
    if (len(ids) != len(set(ids)) or any(not split[name] for name in PARTITIONS)
            or any(not isinstance(uid, str) or "__" not in uid for uid in ids)):
        raise ValueError("The split must contain unique, recording-qualified clip IDs")
    return split


def read_recordings(path: Path) -> dict:
    rows = read_json(path)["recordings"]
    recordings = {}
    for entry in rows:
        uid = entry["recording_uid"]
        if uid in recordings:
            raise ValueError(f"Duplicate recording {uid}")
        recordings[uid] = {
            key: path.parent / entry[key] for key in ("geometry_root", "label_root")
            if key in entry
        }
    return recordings


def field(container, key, shape, kind):
    value = np.asarray(container[key])
    if value.shape != shape or (kind is not None and value.dtype.kind not in kind):
        raise ValueError(f"{key}: expected {shape} and dtype kind {kind}, got {value.shape} {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def mask(container, key, length):
    value = field(container, key, (length,), "bui")
    if not np.isin(value, [0, 1]).all():
        raise ValueError(f"{key}: expected a binary mask")
    return value.astype(bool)


def unique_rows(rows, name):
    entries = {row["sample_id"]: row for row in rows}
    if len(entries) != len(rows):
        raise ValueError(f"Duplicate sample ID in {name}")
    return entries


class ClipDataset(Dataset):
    def __init__(self, options, sample_ids, *, labeled=True):
        self.layout = options
        self.labeled = labeled
        self.recordings = read_recordings(options.manifest) if options.manifest else {}
        self.geometry = {}
        self.labels = {}
        self.layer_paths = {}
        self.items = {}
        self.sample_ids = tuple(sample_ids)
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("Duplicate clip ID")
        for uid in self.sample_ids:
            recording, sample = uid.rsplit("__", 1)
            if any(Path(part).name != part or part in ("", ".", "..")
                   for part in (recording, sample)):
                raise ValueError(f"Invalid clip identifier {uid!r}")
            paths = [
                root / "samples" / f"{uid}.safetensors" for root in options.da3_dir
                if (root / "samples" / f"{uid}.safetensors").is_file()
            ]
            if len(paths) != 1:
                raise ValueError(f"{uid}: expected exactly one DA3 feature file, found {len(paths)}")
            self.layer_paths[uid] = paths[0]
            if labeled and recording not in self.geometry:
                record = self.recordings[recording]
                for dest, key in ((self.geometry, "geometry_root"), (self.labels, "label_root")):
                    path = record[key] / "manifest.jsonl"
                    dest[recording] = unique_rows(read_rows(path), path)

    def __len__(self):
        return len(self.sample_ids)

    def targets(self, uid):
        recording, sample = uid.rsplit("__", 1)
        record = self.labels[recording][sample]["labels"]
        if record.get("contact_threshold_n") != 2.0 or record.get("contact_comparator") != ">":
            raise ValueError(f"{uid}: expected the strict 2 N contact protocol")
        values = np.array([record[key] for key in TARGETS], dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{uid}: invalid scalar targets")
        return values

    def _features(self, uid, recording, sample, timestamps=None):
        cache_path = Path(recording) / f"{sample}.npz"
        with np.load(self.layout.compact_dir / cache_path, allow_pickle=False) as cache:
            length = len(cache["frame_timestamps_ns"])
            time = field(cache, "frame_timestamps_ns", (length,), "iu")
            if length == 0 or (np.diff(time) <= 0).any():
                raise ValueError("Expected strictly increasing frame timestamps")
            if timestamps is not None and not np.array_equal(time, timestamps):
                raise ValueError("DINO frame timestamps differ from geometry")
            if str(cache["sample_id"]) != sample or str(cache["recording_uid"]) != recording:
                raise ValueError("DINO cache has a different clip identity")
            valid = mask(cache, "visual_valid_mask", length)
            tokens = field(cache, "tokens", (length, 5, 1024), "f")
            if str(cache["visual_token_names"][0]) != "cls":
                raise ValueError("First token must be the RGB class token")
            result = {"dino_cls": tokens[:, 0].copy(), "visual_valid": valid}
        with np.load(self.layout.spatial_dir / cache_path, allow_pickle=False) as cache:
            if (str(cache["sample_id"]) != sample or str(cache["recording_uid"]) != recording
                    or not np.array_equal(time, cache["frame_timestamps_ns"])):
                raise ValueError("Spatial DINO cache has a different clip identity or timeline")
            for key in ("contains_gt_metric_scale", "contains_gt_tcp"):
                if key in cache and bool(cache[key]):
                    raise ValueError("DINO cache includes ground-truth geometry")
            result["visual_valid"] &= mask(cache, "visual_valid_mask", length)
            result["dino_mean"] = field(cache, "dino_patch_mean", (length, 1024), "f")
            result["dino_grid"] = field(cache, "dino_grid_7x7", (length, 7, 7, 1024), "f")
        with safe_open(self.layer_paths[uid], framework="pt") as cache:
            meta = cache.metadata() or {}
            if meta.get("sample_uid") != uid:
                raise ValueError("DA3 cache has a different clip identity")
            if not np.array_equal(cache.get_tensor("frame_timestamps_ns").numpy(), time):
                raise ValueError("DA3 cache has a different timeline")
            da3_valid = cache.get_tensor("da3_token_valid_mask")
            if da3_valid.shape != (length,) or not ((da3_valid == 0) | (da3_valid == 1)).all():
                raise ValueError("Invalid DA3 frame mask")
            result["geometry_valid"] = da3_valid.bool() & torch.from_numpy(valid)
            grid = cache.get_tensor("da3_intermediate_grid_7x7")
            if grid.shape != (length, 7, 7, 2048) or not grid.is_floating_point():
                raise ValueError("Invalid DA3 grid")
            if not torch.isfinite(grid).all():
                raise ValueError("Non-finite DA3 feature")
            result["da3_grid"] = grid
        if not result["geometry_valid"].any():
            raise ValueError("Clip has no valid paired RGB and DA3 frames")
        return result, time

    def _labels(self, uid, recording, sample):
        geometry = self.geometry[recording][sample]
        label = self.labels[recording][sample]
        length = int(geometry["counts"]["frames"])
        if length != label["counts"]["aria_human_rgb"]:
            raise ValueError("Geometry and label frame counts differ")
        record = self.recordings[recording]
        with np.load(
            record["geometry_root"] / geometry["paths"]["frames_npz"], allow_pickle=False
        ) as frames:
            timestamps = field(frames, "timestamps_ns", (length,), "iu")
        with np.load(
            record["geometry_root"] / geometry["paths"]["bins_npz"], allow_pickle=False
        ) as bins:
            if not np.array_equal(bins["frame_timestamps_ns"], timestamps):
                raise ValueError("Frame and supervision timestamps differ")
            edges = field(bins, "bin_edges_ns", (length + 1,), "iu")
            if (np.diff(edges) <= 0).any() or edges[0] != label["interaction_start_ns"] or edges[-1] != label["interaction_end_ns"]:
                raise ValueError("Invalid bin edges")
            contact_duration = field(bins, "contact_duration_s", (length,), "f")
            if (contact_duration < 0).any() or (contact_duration > np.diff(edges) * 1e-9 + 1e-5).any():
                raise ValueError("Invalid bin contact durations")
            contact = contact_duration > 0
            vector = bins["contact_force_external_on_rig_mean_common_n"]
            velocity = bins["contact_tcp_velocity_mean_common_m_s"]
            if vector.shape != (length, 3) or velocity.shape != vector.shape:
                raise ValueError("Incorrect force or velocity bin shape")
            mean_valid = contact & np.isfinite(vector).all(1) & np.isfinite(velocity).all(1)
            mean_force = np.linalg.norm(np.where(mean_valid[:, None], vector, 0), axis=1)
            energy = field(bins, "signed_energy_external_on_rig_j", (length,), "f")
            peaks = bins["peak_force_resultant_n"]
        targets = self.targets(uid)
        if not np.isclose(abs(energy.astype(np.float64).sum()), targets[1], rtol=1e-6, atol=1e-7):
            raise ValueError("Bin energies disagree with mechanical work in joules")
        if (peaks.shape != (length,) or np.any(peaks[np.isfinite(peaks)] < 0)
                or (np.isfinite(peaks).all() and not np.isclose(peaks.max(), targets[0], rtol=1e-6))):
            raise ValueError("Bin peaks disagree with peak contact force in newtons")
        return timestamps, {
            "target": targets, "mean_force_target": mean_force.astype(np.float32),
            "mean_force_valid": mean_valid, "contact": contact,
        }

    def load(self, index, *, verify_rgb=False):
        uid = self.sample_ids[index]
        recording, sample = uid.rsplit("__", 1)
        try:
            timestamps, labels = self._labels(uid, recording, sample) if self.labeled else (None, {})
            features, time = self._features(uid, recording, sample, timestamps)
            # The time embedding depends on the RGB clock only.
            t = (time - time[0]) * 1e-9
            cadence = np.diff(t, append=t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 1 / 30))
            timing = np.stack((t / max(t[-1], 1e-6), np.log1p(30 * cadence)), axis=-1)
            if verify_rgb:
                self.verify_rgb(recording, sample, time, self.layout.rgb_root)
            values = {**features, **labels, "time": timing.astype(np.float32)}
            return {
                "uid": uid, "sequence": sample,
                **{key: torch.as_tensor(value) for key, value in values.items()},
            }
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(f"{uid}: {error}") from error

    def verify_rgb(self, recording, sample, timestamps, rgb_root):
        record = self.labels[recording][sample]
        label_root = self.recordings[recording]["label_root"]
        with (label_root / record["paths"]["frames"]).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["timestamp"]) for row in rows] != timestamps.tolist():
            raise ValueError("RGB frame list and cache timestamps differ")
        if rgb_root is not None:
            from PIL import Image

            for row in rows:
                with Image.open(rgb_root / row["rgb_path"]) as frame:
                    frame.load()

    def __getitem__(self, index):
        if not getattr(self.layout, "cache_memory", False):
            return self.load(index)
        if index not in self.items:
            self.items[index] = self.load(index)
        return self.items[index]


def collate(items):
    batch = {"uid": [item["uid"] for item in items], "sequence": [item["sequence"] for item in items]}
    for key in items[0].keys() - batch.keys() - {"target"}:
        batch[key] = pad_sequence([item[key] for item in items], batch_first=True)
    if "target" in items[0]:
        batch["target"] = torch.stack([item["target"] for item in items])
    return batch
