#!/usr/bin/env python3
"""Generate HoI T2 labels from manually confirmed interaction windows.

Version 2 follows the detailed challenge definition supplied by the organizer:

* force is the published gravity/bias-compensated, low-pass-filtered external
  force vector;
* contact is the strict sample-wise condition ``||F|| > 2 N``;
* ``max_force_n`` is the maximum resultant force over the complete manually
  confirmed interaction;
* ``work_j`` is ``abs(integral_contact(F dot v_tool dt))``.  Signed work from
  every disjoint contact run is summed before the single outer absolute value.

The raw dataset is never modified.  Output is built atomically in a separate
directory and includes scalar labels, full-rate synchronized signals, per-frame
nearest-F/T supervision, manifests, and QC plots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


VERSION = "t2-label-v2"
CONTACT_THRESHOLD_N = 2.0
FORCE_COLUMNS = [
    "wrench_ext.force.x_filt",
    "wrench_ext.force.y_filt",
    "wrench_ext.force.z_filt",
]
POSE_V_COLUMNS = [
    "device_linear_velocity_x_device",
    "device_linear_velocity_y_device",
    "device_linear_velocity_z_device",
]
POSE_W_COLUMNS = [
    "angular_velocity_x_device",
    "angular_velocity_y_device",
    "angular_velocity_z_device",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--confirmed-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--recording", required=True, help="Aria recording directory ending in _vrs")
    parser.add_argument("--bag-recording", help="Gripper bag directory; derived from --recording when omitted")
    parser.add_argument("--contact-threshold-n", type=float, default=CONTACT_THRESHOLD_N)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {matrix.shape}")
    return matrix


def calibration_transforms(
    bag_root: Path,
    aria_gripper_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reproduce the public visual-force utility's F/T and tool transforms."""
    bag_calib = bag_root / "calib"
    with (bag_calib / "merged_calibration-camchain.yaml").open(encoding="utf-8") as handle:
        camchain = yaml.safe_load(handle)
    with (bag_calib / "merged_calibration-camchain-imucam.yaml").open(encoding="utf-8") as handle:
        imucam = yaml.safe_load(handle)
    with (bag_calib / "merged_calibration-imu.yaml").open(encoding="utf-8") as handle:
        imu = yaml.safe_load(handle)
    aria_calib_path = aria_gripper_root / "calib" / "calib.json"
    aria_calib = json.loads(aria_calib_path.read_text(encoding="utf-8"))

    t_cam0_imu = load_matrix(imucam["cam0"]["T_cam_imu"])
    t_imu_ft = load_matrix(imu["imu0"]["T_i_s"])
    t_imu_tool = load_matrix(imu["imu0"]["T_i_tool"])
    t_cam0_ft = t_cam0_imu @ t_imu_ft

    t_device_cam_raw = load_matrix(aria_calib["PINHOLE"]["T_device_camera"])
    t_cam2_cam1 = load_matrix(camchain["cam2"]["T_cn_cnm1"])
    t_cam1_cam0 = load_matrix(camchain["cam1"]["T_cn_cnm1"])
    t_cam_raw_cam0 = t_cam2_cam1 @ t_cam1_cam0
    t_device_cam0 = t_device_cam_raw @ t_cam_raw_cam0
    r_device_ft = t_device_cam0[:3, :3] @ t_cam0_ft[:3, :3]

    # This chain maps tool-frame coordinates into the Aria device frame.
    # Its translation is therefore the vector from the device origin to the
    # tool origin, expressed in device coordinates, which is exactly the r in
    # v_tool = v_device + omega x r.
    t_device_tool = t_device_cam0 @ t_cam0_imu @ t_imu_tool
    device_to_tool_offset = t_device_tool[:3, 3]

    # The public visual-force utility inverts T_device_tool and then rotates
    # the inverse translation back without the required minus sign.  Retain
    # that legacy value only for a sensitivity report; it is not used by the
    # canonical v2 label.
    t_tool_device = np.linalg.inv(t_device_tool)
    public_legacy_offset = t_tool_device[:3, :3].T @ t_tool_device[:3, 3]

    determinant = float(np.linalg.det(r_device_ft))
    orthogonality_error = float(np.linalg.norm(r_device_ft.T @ r_device_ft - np.eye(3)))
    if not np.isclose(determinant, 1.0, atol=2e-3) or orthogonality_error > 2e-3:
        raise RuntimeError(
            "Invalid F/T rotation: "
            f"determinant={determinant}, orthogonality_error={orthogonality_error}"
        )
    return (
        r_device_ft,
        device_to_tool_offset,
        public_legacy_offset,
        {
            "rotation_device_from_ft_determinant": determinant,
            "rotation_device_from_ft_orthogonality_error": orthogonality_error,
            "selected_device_to_tool_offset_device_m": device_to_tool_offset.tolist(),
            "selected_tool_offset_method": "translation of T_device_tool",
            "public_legacy_tool_offset_device_m": public_legacy_offset.tolist(),
            "public_legacy_offset_is_diagnostic_only": True,
            "source": (
                "public evaluation_tools_visual_force_estimation.py calibration chain; "
                "tool-offset sign corrected from the composed T_device_tool transform"
            ),
        },
    )


def timestamped_files(path: Path, suffix: str) -> pd.DataFrame:
    rows: list[tuple[int, str]] = []
    for item in path.glob(f"*{suffix}"):
        try:
            rows.append((int(item.stem), str(item)))
        except ValueError:
            continue
    rows.sort()
    if not rows:
        raise RuntimeError(f"No timestamped {suffix} files found in {path}")
    return pd.DataFrame(rows, columns=["timestamp", "path"])


def interpolate_columns(
    source: pd.DataFrame,
    targets: np.ndarray,
    columns: list[str],
    nearest_tolerance_ns: int = 5_000_000,
) -> np.ndarray:
    timestamps_int = source["timestamp"].to_numpy(dtype=np.int64)
    timestamps = timestamps_int.astype(np.float64)
    targets_float = targets.astype(np.float64)
    if targets[0] < timestamps_int[0] or targets[-1] > timestamps_int[-1]:
        raise RuntimeError("F/T timestamps fall outside the gripper-pose interpolation range")
    nearest = nearest_indices(timestamps_int, targets)
    nearest_error_ns = np.abs(timestamps_int[nearest] - targets)
    if int(nearest_error_ns.max()) > nearest_tolerance_ns:
        raise RuntimeError(
            "Local gripper-pose alignment exceeds "
            f"{nearest_tolerance_ns * 1e-6:.3f} ms; "
            f"observed {nearest_error_ns.max() * 1e-6:.3f} ms"
        )
    return np.column_stack(
        [
            np.interp(targets_float, timestamps, source[column].to_numpy(dtype=np.float64))
            for column in columns
        ]
    )


def nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Return nearest source index for each target timestamp."""
    insertion = np.searchsorted(source_timestamps, target_timestamps, side="left")
    right = np.clip(insertion, 0, len(source_timestamps) - 1)
    left = np.clip(insertion - 1, 0, len(source_timestamps) - 1)
    choose_right = (
        np.abs(source_timestamps[right] - target_timestamps)
        < np.abs(source_timestamps[left] - target_timestamps)
    )
    return np.where(choose_right, right, left)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def integrate_runs(values: np.ndarray, time_s: np.ndarray, mask: np.ndarray) -> float:
    """Trapezoid-integrate each contact run without bridging non-contact gaps."""
    total = 0.0
    for start, end in contiguous_runs(mask):
        if end - start >= 2:
            total += float(np.trapezoid(values[start:end], time_s[start:end]))
    return total


def cumulative_contact_integral(values: np.ndarray, time_s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    increments = np.zeros(len(values), dtype=np.float64)
    adjacent_contact = mask[:-1] & mask[1:]
    increments[1:] = (
        0.5
        * (values[:-1] + values[1:])
        * np.diff(time_s)
        * adjacent_contact
    )
    return np.cumsum(increments)


def contact_segment_records(
    timestamps_ns: np.ndarray,
    time_s: np.ndarray,
    contact: np.ndarray,
) -> list[dict]:
    segments = []
    for start, end in contiguous_runs(contact):
        duration_s = float(time_s[end - 1] - time_s[start]) if end - start >= 2 else 0.0
        segments.append(
            {
                "start_ns": int(timestamps_ns[start]),
                "end_ns": int(timestamps_ns[end - 1]),
                "duration_s": duration_s,
                "sample_count": int(end - start),
            }
        )
    return segments


def render_qc(
    sample: dict,
    signals: pd.DataFrame,
    output: Path,
    threshold_n: float,
) -> None:
    time_s = signals["time_s"].to_numpy(dtype=np.float64)
    contact = signals["contact_force_gt"].to_numpy(dtype=bool)
    cumulative_signed = signals["cumulative_work_signed_j"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(time_s, signals["force_resultant_filtered_n"], color="black", linewidth=1.2)
    axes[0].axhline(threshold_n, color="tab:red", linestyle="--", label="strict contact: |F| > 2 N")
    axes[0].set_ylabel("|F| [N]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_s, signals["force_x_filtered_sensor_n"], label="Fx")
    axes[1].plot(time_s, signals["force_y_filtered_sensor_n"], label="Fy")
    axes[1].plot(time_s, signals["force_z_filtered_sensor_n"], label="Fz")
    axes[1].set_ylabel("Sensor force [N]")
    axes[1].legend(loc="upper right", ncols=3)

    axes[2].plot(time_s, signals["power_signed_w"], color="tab:green", label="F_device dot v_tool")
    axes[2].axhline(0.0, color="grey", linewidth=0.8)
    speed_axis = axes[2].twinx()
    speed_axis.plot(
        time_s,
        signals["tool_speed_m_s"],
        color="tab:blue",
        alpha=0.45,
        label="tool speed",
    )
    axes[2].set_ylabel("Power [W]")
    speed_axis.set_ylabel("Speed [m/s]", color="tab:blue")
    axes[2].legend(loc="upper left")

    axes[3].plot(time_s, cumulative_signed, color="tab:purple", label="signed contact work")
    axes[3].axhline(
        sample["labels"]["work_signed_j"],
        color="tab:orange",
        linestyle=":",
        label="final signed integral",
    )
    axes[3].set_ylabel("Cumulative work [J]")
    axes[3].set_xlabel("Time since interaction start [s]")
    axes[3].legend(loc="upper left")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.fill_between(
            time_s,
            0,
            1,
            where=contact,
            transform=axis.get_xaxis_transform(),
            color="tab:red",
            alpha=0.10,
        )
    fig.suptitle(
        f"{sample['sample_id']} | max_force_n={sample['labels']['max_force_n']:.4f} N | "
        f"work_j=abs({sample['labels']['work_signed_j']:.4f})="
        f"{sample['labels']['work_j']:.4f} J | "
        f"contact segments={sample['labels']['contact_segment_count']}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def sorted_confirmed_windows(document: dict) -> list[tuple[str, dict]]:
    def numeric_suffix(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Expected window_<integer> key, got {name!r}") from error

    windows = sorted(document.items(), key=lambda item: numeric_suffix(item[0]))
    required = {"start_ns", "end_ns", "duration_s", "single_word_description", "index"}
    for name, window in windows:
        missing = required - set(window)
        if missing:
            raise RuntimeError(f"{name} is missing fields: {sorted(missing)}")
        if int(window["end_ns"]) <= int(window["start_ns"]):
            raise RuntimeError(f"{name} has an invalid time range")
    return windows


def main() -> None:
    args = parse_args()
    if args.contact_threshold_n != CONTACT_THRESHOLD_N:
        raise ValueError(
            "The official v2 definition is fixed at a strict 2.0 N threshold; "
            "refusing a different --contact-threshold-n"
        )
    if not args.recording.endswith("_vrs"):
        raise ValueError("--recording must name the Aria directory ending in _vrs")

    dataset_root = args.dataset_root.resolve()
    confirmed_json = args.confirmed_json.resolve()
    output_root = args.output_root.resolve()
    bag_recording = args.bag_recording or f"{args.recording.removesuffix('_vrs')}_bag"

    if output_root.exists():
        raise FileExistsError(output_root)

    confirmed_document = json.loads(confirmed_json.read_text(encoding="utf-8"))
    if not isinstance(confirmed_document, dict) or not confirmed_document:
        raise RuntimeError("Confirmed interaction document must be a non-empty JSON object")
    windows = sorted_confirmed_windows(confirmed_document)

    object_actions = [
        (int(window["index"]), str(window["single_word_description"]))
        for _, window in windows
    ]
    if len(object_actions) != len(set(object_actions)):
        raise RuntimeError(
            "Duplicate object/action pairs would collide in sample IDs; "
            "add an interaction index before using this recording"
        )

    record_root = dataset_root / args.scene / "gripper"
    bag_root = record_root / "gripper" / bag_recording
    aria_gripper_root = record_root / "aria_gripper" / args.recording
    aria_human_root = record_root / "aria_human" / args.recording
    wrench_path = (
        bag_root
        / "force_torque"
        / "ft_sensor0"
        / "ft_sensor_readings"
        / "wrench"
        / "data.csv"
    )
    pose_path = (
        aria_gripper_root
        / "slam"
        / "closed_loop_trajectory_aligned"
        / "data.csv"
    )
    rgb_dir = aria_human_root / "camera_rgb"
    depth_dir = aria_human_root / "camera_depth"

    # Published F/T exports occasionally contain duplicate timestamps for
    # successive sensor messages.  Preserve the final emitted filtered state
    # at an identical physical time, rather than synthesizing a value by
    # averaging or re-filtering the raw signal.  This is deliberately applied
    # only to the in-memory label input; the raw dataset remains immutable.
    wrench = pd.read_csv(wrench_path).sort_values("timestamp", kind="stable")
    wrench_rows_before_dedup = int(len(wrench))
    wrench_duplicate_timestamp_values_ns = [
        int(timestamp)
        for timestamp in wrench.loc[
            wrench["timestamp"].duplicated(keep=False), "timestamp"
        ]
        .drop_duplicates()
        .tolist()
    ]
    wrench_duplicate_rows_dropped = int(wrench["timestamp"].duplicated().sum())

    pose = pd.read_csv(pose_path).sort_values("timestamp", kind="stable")
    for column in ["timestamp", *FORCE_COLUMNS]:
        if column not in wrench:
            raise RuntimeError(f"Missing filtered external-force column: {column}")
    for column in ["timestamp", *POSE_V_COLUMNS, *POSE_W_COLUMNS]:
        if column not in pose:
            raise RuntimeError(f"Missing gripper-pose/velocity column: {column}")
    if wrench[["timestamp", *FORCE_COLUMNS]].isna().any().any():
        raise RuntimeError("NaN detected in required F/T fields")
    if pose[["timestamp", *POSE_V_COLUMNS, *POSE_W_COLUMNS]].isna().any().any():
        raise RuntimeError("NaN detected in required gripper-pose fields")
    if wrench_duplicate_rows_dropped:
        wrench = wrench.drop_duplicates(subset="timestamp", keep="last").reset_index(
            drop=True
        )
    if not wrench["timestamp"].is_monotonic_increasing or wrench["timestamp"].duplicated().any():
        raise RuntimeError("F/T timestamps must be strictly increasing")
    if not pose["timestamp"].is_monotonic_increasing or pose["timestamp"].duplicated().any():
        raise RuntimeError("Gripper-pose timestamps must be strictly increasing")

    rgb = timestamped_files(rgb_dir, ".jpg")
    depth = timestamped_files(depth_dir, ".npy")
    if not np.array_equal(rgb["timestamp"].to_numpy(), depth["timestamp"].to_numpy()):
        raise RuntimeError("aria_human RGB and derived depth timestamps are not exactly paired")
    depth_lookup = dict(zip(depth["timestamp"], depth["path"]))

    (
        r_device_ft,
        device_to_tool_offset,
        public_legacy_tool_offset,
        transform_metadata,
    ) = calibration_transforms(
        bag_root,
        aria_gripper_root,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    samples: list[dict] = []
    label_rows: list[dict] = []
    tool_offset_diagnostic_rows: list[dict] = []

    try:
        for final_window_name, window in windows:
            object_index = int(window["index"])
            action = str(window["single_word_description"])
            sample_id = f"{args.scene}_obj{object_index:02d}_{action}"
            sample_dir = temp_root / "samples" / sample_id
            sample_dir.mkdir(parents=True)

            start_ns = int(window["start_ns"])
            end_ns = int(window["end_ns"])
            selected = wrench[
                (wrench["timestamp"] >= start_ns)
                & (wrench["timestamp"] <= end_ns)
            ].copy()
            if len(selected) < 2:
                raise RuntimeError(f"Insufficient F/T samples for {sample_id}")

            timestamps_ns = selected["timestamp"].to_numpy(dtype=np.int64)
            if np.any(np.diff(timestamps_ns) <= 0):
                raise RuntimeError(f"Non-increasing F/T timestamps in {sample_id}")
            time_s = (timestamps_ns - timestamps_ns[0]) * 1e-9

            force_sensor = selected[FORCE_COLUMNS].to_numpy(dtype=np.float64)
            force_resultant = np.linalg.norm(force_sensor, axis=1)
            contact = force_resultant > CONTACT_THRESHOLD_N
            force_device = (r_device_ft @ force_sensor.T).T

            velocity_device = interpolate_columns(pose, timestamps_ns, POSE_V_COLUMNS)
            angular_velocity_device = interpolate_columns(pose, timestamps_ns, POSE_W_COLUMNS)
            pose_timestamps_ns = pose["timestamp"].to_numpy(dtype=np.int64)
            nearest_pose = nearest_indices(pose_timestamps_ns, timestamps_ns)
            pose_alignment_error_ms = (
                np.abs(pose_timestamps_ns[nearest_pose] - timestamps_ns) * 1e-6
            )
            velocity_tool = velocity_device + np.cross(
                angular_velocity_device,
                device_to_tool_offset,
            )
            velocity_tool_public_legacy = velocity_device + np.cross(
                angular_velocity_device,
                public_legacy_tool_offset,
            )
            tool_speed = np.linalg.norm(velocity_tool, axis=1)
            power_signed = np.einsum("ij,ij->i", force_device, velocity_tool)
            power_public_legacy = np.einsum(
                "ij,ij->i",
                force_device,
                velocity_tool_public_legacy,
            )
            power_contact_gated = np.where(contact, power_signed, 0.0)
            cumulative_signed = cumulative_contact_integral(
                power_signed,
                time_s,
                contact,
            )

            work_signed_j = integrate_runs(power_signed, time_s, contact)
            work_j = abs(work_signed_j)
            work_signed_public_legacy_j = integrate_runs(
                power_public_legacy,
                time_s,
                contact,
            )
            work_public_legacy_j = abs(work_signed_public_legacy_j)
            peak_index = int(np.argmax(force_resultant))
            contact_segments = contact_segment_records(timestamps_ns, time_s, contact)
            contact_indices = np.flatnonzero(contact)
            contact_duration_s = float(
                sum(segment["duration_s"] for segment in contact_segments)
            )

            labels = {
                "label_version": VERSION,
                "max_force_n": float(force_resultant[peak_index]),
                "peak_force_timestamp_ns": int(timestamps_ns[peak_index]),
                "work_j": float(work_j),
                "work_signed_j": float(work_signed_j),
                "contact_threshold_n": CONTACT_THRESHOLD_N,
                "contact_comparator": ">",
                "contact_sample_count": int(contact.sum()),
                "contact_segment_count": int(len(contact_segments)),
                "contact_duration_s": contact_duration_s,
                "median_ft_dt_ms": float(np.median(np.diff(timestamps_ns)) * 1e-6),
                "max_ft_dt_ms": float(np.max(np.diff(timestamps_ns)) * 1e-6),
                "max_pose_alignment_error_ms": float(pose_alignment_error_ms.max()),
                "contact_first_ns": (
                    None if not len(contact_indices) else int(timestamps_ns[contact_indices[0]])
                ),
                "contact_last_ns": (
                    None if not len(contact_indices) else int(timestamps_ns[contact_indices[-1]])
                ),
                "max_force_contact_n": (
                    None if not len(contact_indices) else float(force_resultant[contact].max())
                ),
            }

            signals = pd.DataFrame(
                {
                    "timestamp": timestamps_ns,
                    "time_s": time_s,
                    "force_x_filtered_sensor_n": force_sensor[:, 0],
                    "force_y_filtered_sensor_n": force_sensor[:, 1],
                    "force_z_filtered_sensor_n": force_sensor[:, 2],
                    "force_resultant_filtered_n": force_resultant,
                    "force_x_device_n": force_device[:, 0],
                    "force_y_device_n": force_device[:, 1],
                    "force_z_device_n": force_device[:, 2],
                    "tool_velocity_x_device_m_s": velocity_tool[:, 0],
                    "tool_velocity_y_device_m_s": velocity_tool[:, 1],
                    "tool_velocity_z_device_m_s": velocity_tool[:, 2],
                    "tool_speed_m_s": tool_speed,
                    "nearest_pose_alignment_error_ms": pose_alignment_error_ms,
                    "contact_force_gt": contact.astype(np.uint8),
                    "power_signed_w": power_signed,
                    "power_contact_gated_w": power_contact_gated,
                    "cumulative_work_signed_j": cumulative_signed,
                }
            )
            signals.to_csv(sample_dir / "signals.csv", index=False)

            frames = rgb[
                (rgb["timestamp"] >= start_ns) & (rgb["timestamp"] <= end_ns)
            ].copy()
            if frames.empty:
                raise RuntimeError(f"No aria_human RGB frames for {sample_id}")
            frames.rename(columns={"path": "rgb_path"}, inplace=True)
            frames["depth_path"] = frames["timestamp"].map(depth_lookup)
            if frames["depth_path"].isna().any():
                raise RuntimeError(f"Unpaired derived-depth frames in {sample_id}")
            frame_timestamps = frames["timestamp"].to_numpy(dtype=np.int64)
            nearest = nearest_indices(timestamps_ns, frame_timestamps)
            frames["nearest_ft_timestamp_ns"] = timestamps_ns[nearest]
            frames["nearest_ft_time_error_ms"] = (
                np.abs(frame_timestamps - timestamps_ns[nearest]) * 1e-6
            )
            frames["nearest_ft_valid_5ms"] = (
                frames["nearest_ft_time_error_ms"] <= 5.0
            ).astype(np.uint8)
            frames["nearest_force_x_filtered_sensor_n"] = force_sensor[nearest, 0]
            frames["nearest_force_y_filtered_sensor_n"] = force_sensor[nearest, 1]
            frames["nearest_force_z_filtered_sensor_n"] = force_sensor[nearest, 2]
            frames["nearest_force_resultant_filtered_n"] = force_resultant[nearest]
            frames["nearest_contact_force_gt"] = contact[nearest].astype(np.uint8)
            frames["nearest_power_signed_w"] = power_signed[nearest]
            frames["nearest_tool_speed_m_s"] = tool_speed[nearest]
            frames["rgb_path"] = frames["rgb_path"].map(
                lambda value: str(Path(value).relative_to(dataset_root))
            )
            frames["depth_path"] = frames["depth_path"].map(
                lambda value: str(Path(value).relative_to(dataset_root))
            )
            frames.to_csv(sample_dir / "frames.csv", index=False)

            sample = {
                "sample_id": sample_id,
                "scene": args.scene,
                "recording": args.recording,
                "bag_recording": bag_recording,
                "object_index": object_index,
                "action": action,
                "final_window_name": final_window_name,
                "interaction_start_ns": start_ns,
                "interaction_end_ns": end_ns,
                "interaction_duration_s": float(window["duration_s"]),
                "counts": {
                    "aria_human_rgb": int(len(frames)),
                    "aria_human_derived_depth": int(len(frames)),
                    "force_torque": int(len(signals)),
                    "frames_with_valid_nearest_ft_5ms": int(
                        frames["nearest_ft_valid_5ms"].sum()
                    ),
                },
                "paths": {
                    "frames": f"samples/{sample_id}/frames.csv",
                    "signals": f"samples/{sample_id}/signals.csv",
                    "qc_plot": f"samples/{sample_id}/qc.png",
                },
                "labels": labels,
                "contact_segments_ns": contact_segments,
            }
            atomic_json(sample_dir / "manifest.json", sample)
            render_qc(sample, signals, sample_dir / "qc.png", CONTACT_THRESHOLD_N)
            samples.append(sample)
            label_rows.append(
                {
                    "sample_id": sample_id,
                    "scene": args.scene,
                    "recording": args.recording,
                    "object_index": object_index,
                    "action": action,
                    "final_window_name": final_window_name,
                    "interaction_start_ns": start_ns,
                    "interaction_end_ns": end_ns,
                    **labels,
                }
            )
            tool_offset_diagnostic_rows.append(
                {
                    "sample_id": sample_id,
                    "selected_offset_work_signed_j": work_signed_j,
                    "selected_offset_work_j": work_j,
                    "public_legacy_offset_work_signed_j": work_signed_public_legacy_j,
                    "public_legacy_offset_work_j": work_public_legacy_j,
                    "legacy_minus_selected_work_j": work_public_legacy_j - work_j,
                }
            )

        with (temp_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample) + "\n")
        pd.DataFrame(label_rows).to_csv(temp_root / "labels.csv", index=False)
        diagnostics_root = temp_root / "diagnostics"
        diagnostics_root.mkdir()
        pd.DataFrame(tool_offset_diagnostic_rows).to_csv(
            diagnostics_root / "tool_offset_sensitivity.csv",
            index=False,
        )

        label_config = {
            "version": VERSION,
            "status": "definition_aligned_candidate",
            "official_definition_implemented": {
                "force": "published gravity/bias-compensated low-pass-filtered external force",
                "contact": "strict filtered resultant force > 2.0 N",
                "max_force_n": "maximum resultant force over full confirmed interaction",
                "work_j": "absolute value after summing signed F dot v integrals over all contact runs",
            },
            "force": {
                "source_columns": FORCE_COLUMNS,
                "additional_baseline_subtraction": False,
                "resultant_formula": "sqrt(Fx^2 + Fy^2 + Fz^2)",
            },
            "contact": {
                "threshold_n": CONTACT_THRESHOLD_N,
                "comparator": ">",
                "hysteresis": False,
                "minimum_duration_filter": False,
                "gap_merge": False,
                "padding": False,
                "trigger_used": False,
                "motor_state_used": False,
                "tactile_used": False,
            },
            "work": {
                "formula": "abs(sum_over_contact_runs(trapezoid(F_device dot v_tool_device, t)))",
                "absolute_value_position": "once, after summing all signed contact-run integrals",
                "force_sign_correction": False,
                "torque_term_included": False,
                "tool_velocity_formula": "v_device + omega_device cross r_device_to_tool",
                "integration_time_unit": "seconds converted from nanosecond timestamps",
                "threshold_crossing_interpolation": False,
            },
            "frame_alignment": {
                "method": "nearest F/T sample to each aria_human RGB timestamp",
                "full_rate_ground_truth": "samples/*/signals.csv",
            },
            "pose_alignment": {
                "method": "linear interpolation on the common processed timestamp in nanoseconds",
                "nearest_sample_tolerance_ms": 5.0,
            },
            "transforms": transform_metadata,
            "inputs": {
                "dataset_root": str(dataset_root),
                "confirmed_json": str(confirmed_json),
                "wrench_csv": str(wrench_path),
                "pose_csv": str(pose_path),
                "aria_human_rgb": str(rgb_dir),
                "aria_human_derived_depth": str(depth_dir),
            },
            "input_timestamp_qc": {
                "wrench_rows_before_dedup": wrench_rows_before_dedup,
                "wrench_rows_after_dedup": int(len(wrench)),
                "wrench_duplicate_rows_dropped": wrench_duplicate_rows_dropped,
                "wrench_duplicate_timestamp_values_ns": wrench_duplicate_timestamp_values_ns,
                "wrench_duplicate_policy": "stable_sort_then_keep_last",
            },
            "source_sha256": {
                "confirmed_json": sha256_file(confirmed_json),
                "wrench_csv": sha256_file(wrench_path),
                "pose_csv": sha256_file(pose_path),
                "generator": sha256_file(Path(__file__).resolve()),
            },
            "note": (
                "The challenge's hidden label-generation implementation remains private. "
                "This version implements the organizer's detailed written definition without "
                "extra contact smoothing or sensor baselining."
            ),
        }
        atomic_json(temp_root / "label_config.json", label_config)
        (temp_root / "README.md").write_text(
            "# HoI T2 definition-aligned labels v2\n\n"
            "This directory contains derived labels for manually confirmed interaction windows. "
            "The raw dataset is referenced but never modified.\n\n"
            "Canonical targets:\n\n"
            "- `max_force_n = max_t ||F_filtered(t)||` over the full confirmed interaction.\n"
            "- `contact(t) = 1[||F_filtered(t)|| > 2.0 N]` with no trigger union, "
            "hysteresis, gap merging, minimum-duration filtering, or padding.\n"
            "- `work_j = abs(sum of signed trapezoid integrals of F_device dot v_tool "
            "over every disjoint contact run)`.\n\n"
            "`labels.csv` contains scalar targets. `samples/*/signals.csv` retains the "
            "full F/T-rate label chain, while `frames.csv` provides explicit nearest-F/T "
            "supervision for every aria_human RGB frame; use `nearest_ft_valid_5ms` as "
            "the dense-supervision validity mask. The depth paths refer to the "
            "dataset's RGB-derived relative depth arrays, not metric Aria depth. "
            "`diagnostics/tool_offset_sensitivity.csv` compares the selected physically "
            "direct device-to-tool translation with the opposite-sign legacy public-code "
            "offset; only the selected-offset values enter `labels.csv`.\n",
            encoding="utf-8",
        )
        temp_root.replace(output_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    print(f"DONE output={output_root} samples={len(samples)}")
    print("contact_rule=filtered_resultant_n > 2.0")
    print("work_rule=abs(sum_signed_contact_integrals)")


if __name__ == "__main__":
    main()
