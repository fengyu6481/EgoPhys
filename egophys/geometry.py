#!/usr/bin/env python3
"""Generate synchronized common-frame geometry for HoI T2 interaction windows.

The generator consumes the already validated ``t2-label-v2`` artifacts and the
two aligned Aria trajectories. It never changes the raw dataset or v2 labels.

For every interaction, the common frame is the rectified ``aria_human`` RGB
camera frame at the middle RGB timestamp. The public transform convention is
``T_A_B`` (B coordinates to A coordinates).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry_common import (
    PoseTrajectory,
    atomic_json,
    expand_pose_columns,
    frame_centered_edges,
    invert_transforms,
    load_matrix,
    rotate_vectors,
    sha256_file,
    sliding_window_starts,
    transform_points,
)


VERSION = "geometry-v1"
README_MARKER = "# HoI T2 common-frame geometry v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--t2-label-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--canonical-size", type=int, default=448)
    parser.add_argument("--da3-window", type=int, default=16)
    parser.add_argument("--da3-stride", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def batched_left_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", left, right)


def append_xyz(output: dict[str, np.ndarray], prefix: str, values: np.ndarray) -> None:
    for axis, column in zip("xyz", values.T):
        output[f"{prefix}_{axis}"] = column


def append_uv(output: dict[str, np.ndarray], prefix: str, values: np.ndarray) -> None:
    output[f"{prefix}_u"] = values[:, 0]
    output[f"{prefix}_v"] = values[:, 1]


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    projected = (intrinsics @ points_camera.T).T
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    valid = np.abs(projected[:, 2]) > 1e-12
    uv[valid] = projected[valid, :2] / projected[valid, 2:3]
    return uv


def integrate_linear_to_bins(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    edges_ns: np.ndarray,
    active_samples: np.ndarray | None = None,
) -> np.ndarray:
    """Exactly split a piecewise-linear trapezoid integral across time bins.

    A segment contributes only when both endpoint samples are active. This
    matches the v2 label's disjoint-contact-run integration and never bridges a
    non-contact gap.
    """
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    edges_ns = np.asarray(edges_ns, dtype=np.int64)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    if active_samples is None:
        active_samples = np.ones(len(timestamps_ns), dtype=bool)
    else:
        active_samples = np.asarray(active_samples, dtype=bool)
    result = np.zeros((len(edges_ns) - 1, values.shape[1]), dtype=np.float64)

    for index in range(len(timestamps_ns) - 1):
        if not (active_samples[index] and active_samples[index + 1]):
            continue
        t0 = int(timestamps_ns[index])
        t1 = int(timestamps_ns[index + 1])
        if t1 <= t0:
            raise ValueError("Signal timestamps must be strictly increasing")
        overlap_start = max(t0, int(edges_ns[0]))
        overlap_end = min(t1, int(edges_ns[-1]))
        if overlap_end <= overlap_start:
            continue
        first_bin = int(np.searchsorted(edges_ns, overlap_start, side="right") - 1)
        first_bin = min(max(first_bin, 0), len(result) - 1)
        bin_index = first_bin
        while bin_index < len(result):
            left = max(overlap_start, int(edges_ns[bin_index]))
            right = min(overlap_end, int(edges_ns[bin_index + 1]))
            if right > left:
                fraction_left = (left - t0) / (t1 - t0)
                fraction_right = (right - t0) / (t1 - t0)
                value_left = (
                    values[index]
                    + fraction_left * (values[index + 1] - values[index])
                )
                value_right = (
                    values[index]
                    + fraction_right * (values[index + 1] - values[index])
                )
                result[bin_index] += (
                    0.5 * (value_left + value_right) * ((right - left) * 1e-9)
                )
            if int(edges_ns[bin_index + 1]) >= overlap_end:
                break
            bin_index += 1
    return result[:, 0] if squeeze else result


def sample_statistics_by_bin(
    values: np.ndarray,
    signal_bin_index: np.ndarray,
    bin_count: int,
    selection: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    if selection is None:
        selection = np.ones(len(values), dtype=bool)
    means = np.full((bin_count, values.shape[1]), np.nan, dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int64)
    for bin_index in range(bin_count):
        mask = selection & (signal_bin_index == bin_index)
        counts[bin_index] = int(mask.sum())
        if counts[bin_index]:
            means[bin_index] = values[mask].mean(axis=0)
    if squeeze:
        means = means[:, 0]
    return means, counts


def peak_by_bin(
    values: np.ndarray, signal_bin_index: np.ndarray, bin_count: int
) -> np.ndarray:
    peaks = np.full(bin_count, np.nan, dtype=np.float64)
    for bin_index in range(bin_count):
        selected = values[signal_bin_index == bin_index]
        if len(selected):
            peaks[bin_index] = float(np.max(selected))
    return peaks


def vector_at_scalar_peak_by_bin(
    scalars: np.ndarray,
    vectors: np.ndarray,
    timestamps_ns: np.ndarray,
    signal_bin_index: np.ndarray,
    bin_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_vectors = np.full((bin_count, vectors.shape[1]), np.nan, dtype=np.float64)
    selected_timestamps = np.full(bin_count, -1, dtype=np.int64)
    for bin_index in range(bin_count):
        indices = np.flatnonzero(signal_bin_index == bin_index)
        if not len(indices):
            continue
        selected = int(indices[np.argmax(scalars[indices])])
        selected_vectors[bin_index] = vectors[selected]
        selected_timestamps[bin_index] = timestamps_ns[selected]
    return selected_vectors, selected_timestamps


def window_arrays(
    frame_timestamps: np.ndarray,
    rgb_paths: list[str],
    transforms_world_camera: np.ndarray,
    intrinsics_canonical: np.ndarray,
    window_length: int,
    stride: int,
) -> tuple[dict[str, np.ndarray], list[dict], np.ndarray]:
    starts = sliding_window_starts(len(frame_timestamps), window_length, stride)
    frame_indices = np.stack(
        [np.arange(start, start + window_length, dtype=np.int64) for start in starts]
    )
    timestamps = frame_timestamps[frame_indices]
    reference_local_index = window_length // 2
    transforms_reference_camera = []
    extrinsics_camera_reference = []
    camera_centers_reference = []
    trajectory_rms_radius_m = []
    baseline_m = []
    records = []
    membership = np.zeros(len(frame_timestamps), dtype=np.int64)

    reference_frame_indices = frame_indices[:, reference_local_index]
    owner_window_for_frame = np.empty(len(frame_timestamps), dtype=np.int64)
    for frame_index in range(len(frame_timestamps)):
        containing = np.flatnonzero(
            (frame_indices[:, 0] <= frame_index)
            & (frame_indices[:, -1] >= frame_index)
        )
        if not len(containing):
            raise RuntimeError(f"DA3 windows do not cover frame {frame_index}")
        owner_window_for_frame[frame_index] = containing[
            np.argmin(np.abs(reference_frame_indices[containing] - frame_index))
        ]
    owned_frame_mask = np.zeros_like(frame_indices, dtype=np.uint8)

    for window_id, indices in enumerate(frame_indices):
        membership[indices] += 1
        owned_frame_mask[window_id] = (
            owner_window_for_frame[indices] == window_id
        ).astype(np.uint8)
        reference_frame_index = int(indices[reference_local_index])
        transform_reference_world = np.linalg.inv(
            transforms_world_camera[reference_frame_index]
        )
        local = batched_left_multiply(
            transform_reference_world, transforms_world_camera[indices]
        )
        w2c = invert_transforms(local)
        centers = local[:, :3, 3]
        radius = float(np.sqrt(np.mean(np.sum(centers * centers, axis=1))))
        baseline = float(np.max(np.linalg.norm(centers[:, None] - centers[None], axis=2)))
        transforms_reference_camera.append(local)
        extrinsics_camera_reference.append(w2c)
        camera_centers_reference.append(centers)
        trajectory_rms_radius_m.append(radius)
        baseline_m.append(baseline)
        records.append(
            {
                "window_id": window_id,
                "start_frame_index": int(indices[0]),
                "end_frame_index_inclusive": int(indices[-1]),
                "reference_local_index": reference_local_index,
                "reference_frame_index": reference_frame_index,
                "frame_indices": indices.tolist(),
                "frame_timestamps_ns": timestamps[window_id].tolist(),
                "rgb_paths": [rgb_paths[index] for index in indices],
                "gt_trajectory_rms_radius_m": radius,
                "gt_max_camera_center_baseline_m": baseline,
                "gt_scale_observable_mask_0p03m": int(baseline >= 0.03),
                "owned_local_frame_indices": np.flatnonzero(
                    owned_frame_mask[window_id]
                ).tolist(),
                "metric_scale_multiplier_status": "pending_da3_prediction",
            }
        )

    arrays = {
        "frame_indices": frame_indices,
        "frame_timestamps_ns": timestamps,
        "reference_local_index": np.asarray(reference_local_index, dtype=np.int64),
        "K_canonical": np.repeat(
            intrinsics_canonical[None, None],
            len(starts) * window_length,
            axis=0,
        ).reshape(len(starts), window_length, 3, 3),
        "T_window_reference_camera": np.stack(transforms_reference_camera),
        "E_camera_window_reference_w2c": np.stack(extrinsics_camera_reference),
        "gt_camera_centers_window_reference_m": np.stack(camera_centers_reference),
        "gt_trajectory_rms_radius_m": np.asarray(
            trajectory_rms_radius_m, dtype=np.float64
        ),
        "gt_max_camera_center_baseline_m": np.asarray(
            baseline_m, dtype=np.float64
        ),
        "gt_scale_observable_mask_0p03m": (
            np.asarray(baseline_m, dtype=np.float64) >= 0.03
        ).astype(np.uint8),
        "owned_frame_mask": owned_frame_mask,
        "owner_window_for_frame": owner_window_for_frame,
    }
    return arrays, records, membership


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    t2_root = args.t2_label_root.resolve()
    output_root = args.output_root.resolve()
    if args.canonical_size <= 0 or args.canonical_size % 112:
        raise ValueError("--canonical-size must be a positive multiple of lcm(14,16)=112")
    if args.da3_window < 2 or args.da3_stride < 1:
        raise ValueError("Invalid DA3 window/stride")
    if not args.recording.endswith("_vrs"):
        raise ValueError("--recording must end in _vrs")

    if output_root.exists():
        raise FileExistsError(output_root)

    source_manifest_path = t2_root / "manifest.jsonl"
    source_config_path = t2_root / "label_config.json"
    source_manifest = read_jsonl(source_manifest_path)
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if source_config.get("version") != "t2-label-v2":
        raise ValueError("geometry-v1 requires t2-label-v2")
    tool_offset_device = np.asarray(
        source_config["transforms"]["selected_device_to_tool_offset_device_m"],
        dtype=np.float64,
    )
    if tool_offset_device.shape != (3,):
        raise ValueError("Invalid device-to-TCP offset")

    record_root = dataset_root / args.scene / "gripper"
    human_root = record_root / "aria_human" / args.recording
    gripper_root = record_root / "aria_gripper" / args.recording
    human_trajectory_path = (
        human_root / "slam" / "closed_loop_trajectory_aligned" / "data.csv"
    )
    gripper_trajectory_path = (
        gripper_root / "slam" / "closed_loop_trajectory_aligned" / "data.csv"
    )
    human_calibration_path = human_root / "calib" / "calib.json"
    human_trajectory = PoseTrajectory(
        pd.read_csv(human_trajectory_path), "aria_human"
    )
    gripper_trajectory = PoseTrajectory(
        pd.read_csv(gripper_trajectory_path), "aria_gripper"
    )
    human_calibration = json.loads(human_calibration_path.read_text(encoding="utf-8"))
    pinhole = human_calibration["PINHOLE"]
    intrinsics_original = np.asarray(pinhole["K"], dtype=np.float64)
    original_height = int(pinhole["h"])
    original_width = int(pinhole["w"])
    if original_height != original_width:
        raise ValueError("geometry-v1 canonical transform currently requires square RGB")
    transform_human_device_camera = (
        load_matrix(pinhole["T_device_camera"])
        @ load_matrix(pinhole["pinhole_T_device_camera"])
    )
    scale = args.canonical_size / original_width
    # Pixel-center preserving resize:
    # u_model = (u_original + 0.5) * scale - 0.5.
    center_offset = 0.5 * (scale - 1.0)
    transform_original_canonical = np.asarray(
        [
            [scale, 0.0, center_offset],
            [0.0, scale, center_offset],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    intrinsics_canonical = transform_original_canonical @ intrinsics_original

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    output_records: list[dict] = []
    summary_rows: list[dict] = []
    try:
        for source_record in source_manifest:
            sample_id = source_record["sample_id"]
            source_sample_root = t2_root / "samples" / sample_id
            frame_source_path = t2_root / source_record["paths"]["frames"]
            signal_source_path = t2_root / source_record["paths"]["signals"]
            frames_source = pd.read_csv(frame_source_path)
            signals_source = pd.read_csv(signal_source_path)
            frame_timestamps = frames_source["timestamp"].to_numpy(dtype=np.int64)
            signal_timestamps = signals_source["timestamp"].to_numpy(dtype=np.int64)
            start_ns = int(source_record["interaction_start_ns"])
            end_ns = int(source_record["interaction_end_ns"])
            if np.any(np.diff(frame_timestamps) <= 0) or np.any(
                np.diff(signal_timestamps) <= 0
            ):
                raise ValueError(f"{sample_id}: non-increasing timestamps")

            # Poses at RGB timestamps.
            (
                transforms_world_human_device,
                _,
                _,
                human_pose_error_ms,
            ) = human_trajectory.interpolate(frame_timestamps)
            transforms_world_human_camera = (
                transforms_world_human_device @ transform_human_device_camera
            )
            (
                transforms_world_gripper_device,
                gripper_linear_velocity_device,
                gripper_angular_velocity_device,
                gripper_pose_error_ms,
            ) = gripper_trajectory.interpolate(frame_timestamps)

            tcp_device = np.repeat(tool_offset_device[None], len(frame_timestamps), axis=0)
            tcp_position_world = transform_points(
                transforms_world_gripper_device, tcp_device
            )
            tcp_velocity_device = gripper_linear_velocity_device + np.cross(
                gripper_angular_velocity_device, tool_offset_device
            )
            tcp_velocity_world = rotate_vectors(
                transforms_world_gripper_device, tcp_velocity_device
            )

            reference_frame_index = len(frame_timestamps) // 2
            transform_common_world = np.linalg.inv(
                transforms_world_human_camera[reference_frame_index]
            )
            transforms_common_human_camera = batched_left_multiply(
                transform_common_world, transforms_world_human_camera
            )
            transforms_common_gripper_device = batched_left_multiply(
                transform_common_world, transforms_world_gripper_device
            )
            tcp_position_common = (
                (transform_common_world[:3, :3] @ tcp_position_world.T).T
                + transform_common_world[:3, 3]
            )
            tcp_velocity_common = (
                transform_common_world[:3, :3] @ tcp_velocity_world.T
            ).T

            transforms_human_camera_world = invert_transforms(
                transforms_world_human_camera
            )
            tcp_position_human_camera = transform_points(
                transforms_human_camera_world, tcp_position_world
            )
            uv_original = project_points(
                tcp_position_human_camera, intrinsics_original
            )
            uv_canonical = (
                transform_original_canonical
                @ np.column_stack((uv_original, np.ones(len(uv_original)))).T
            ).T[:, :2]
            in_front = tcp_position_human_camera[:, 2] > 1e-4
            in_bounds = (
                (uv_original[:, 0] >= 0.0)
                & (uv_original[:, 0] < original_width)
                & (uv_original[:, 1] >= 0.0)
                & (uv_original[:, 1] < original_height)
            )
            pose_sync_valid = (
                (human_pose_error_ms <= 5.0) & (gripper_pose_error_ms <= 5.0)
            )
            visibility_frustum = in_front & in_bounds & pose_sync_valid

            # F/T-rate force, velocity, pose and TCP in the same fixed frame.
            (
                transforms_world_gripper_signal,
                _,
                _,
                gripper_signal_pose_error_ms,
            ) = gripper_trajectory.interpolate(signal_timestamps)
            force_device = signals_source[
                [
                    "force_x_device_n",
                    "force_y_device_n",
                    "force_z_device_n",
                ]
            ].to_numpy(dtype=np.float64)
            velocity_device = signals_source[
                [
                    "tool_velocity_x_device_m_s",
                    "tool_velocity_y_device_m_s",
                    "tool_velocity_z_device_m_s",
                ]
            ].to_numpy(dtype=np.float64)
            # The published external wrench is the external/object force acting
            # on the manipulation rig. The opposite vector is included as an
            # explicitly named candidate for "force transferred to object";
            # their resultant and final abs(work) are identical.
            force_world = rotate_vectors(transforms_world_gripper_signal, force_device)
            velocity_world = rotate_vectors(
                transforms_world_gripper_signal, velocity_device
            )
            rotation_common_world = transform_common_world[:3, :3]
            force_common = (rotation_common_world @ force_world.T).T
            velocity_common = (rotation_common_world @ velocity_world.T).T
            tcp_signal_world = transform_points(
                transforms_world_gripper_signal,
                np.repeat(tool_offset_device[None], len(signal_timestamps), axis=0),
            )
            tcp_signal_common = (
                (rotation_common_world @ tcp_signal_world.T).T
                + transform_common_world[:3, 3]
            )
            power_common = np.einsum("ij,ij->i", force_common, velocity_common)
            power_source = signals_source["power_signed_w"].to_numpy(dtype=np.float64)
            max_power_invariance_error = float(
                np.max(np.abs(power_common - power_source))
            )
            if max_power_invariance_error > 1e-8:
                raise RuntimeError(
                    f"{sample_id}: force/velocity common-frame dot-product drift "
                    f"{max_power_invariance_error}"
                )
            contact = signals_source["contact_force_gt"].to_numpy(dtype=np.uint8).astype(
                bool
            )
            resultant = signals_source[
                "force_resultant_filtered_n"
            ].to_numpy(dtype=np.float64)

            # One exact time bin per RGB frame.
            edges_ns = frame_centered_edges(frame_timestamps, start_ns, end_ns)
            bin_count = len(frame_timestamps)
            signal_bin_index = (
                np.searchsorted(edges_ns, signal_timestamps, side="right") - 1
            )
            signal_bin_index = np.clip(signal_bin_index, 0, bin_count - 1)
            peak_resultant = peak_by_bin(
                resultant, signal_bin_index, bin_count
            )
            force_at_peak_common, peak_signal_timestamp_ns = (
                vector_at_scalar_peak_by_bin(
                    resultant,
                    force_common,
                    signal_timestamps,
                    signal_bin_index,
                    bin_count,
                )
            )
            force_mean_contact, contact_sample_count = sample_statistics_by_bin(
                force_common, signal_bin_index, bin_count, contact
            )
            velocity_mean_contact, _ = sample_statistics_by_bin(
                velocity_common, signal_bin_index, bin_count, contact
            )
            _, all_sample_count = sample_statistics_by_bin(
                resultant, signal_bin_index, bin_count
            )
            signed_energy = integrate_linear_to_bins(
                signal_timestamps, power_common, edges_ns, contact
            )
            contact_duration = integrate_linear_to_bins(
                signal_timestamps,
                np.ones(len(signal_timestamps), dtype=np.float64),
                edges_ns,
                contact,
            )
            speed = np.linalg.norm(velocity_common, axis=1)
            path_length_contact = integrate_linear_to_bins(
                signal_timestamps, speed, edges_ns, contact
            )
            path_length_all = integrate_linear_to_bins(
                signal_timestamps, speed, edges_ns
            )
            bin_duration = np.diff(edges_ns) * 1e-9
            contact_occupancy_duration = np.divide(
                contact_duration,
                bin_duration,
                out=np.zeros_like(contact_duration),
                where=bin_duration > 0,
            )
            contact_occupancy_samples = np.divide(
                contact_sample_count,
                all_sample_count,
                out=np.zeros(bin_count, dtype=np.float64),
                where=all_sample_count > 0,
            )
            peak_global = float(np.nanmax(peak_resultant))
            expected_peak = float(source_record["labels"]["max_force_n"])
            energy_global = float(np.sum(signed_energy))
            expected_signed_energy = float(source_record["labels"]["work_signed_j"])
            if not np.isclose(peak_global, expected_peak, atol=1e-10, rtol=0.0):
                raise RuntimeError(f"{sample_id}: bin peak does not preserve label")
            if not np.isclose(
                energy_global, expected_signed_energy, atol=1e-10, rtol=0.0
            ):
                raise RuntimeError(
                    f"{sample_id}: bin energy {energy_global} does not preserve "
                    f"source {expected_signed_energy}"
                )

            sample_root = temporary_root / "samples" / sample_id
            sample_root.mkdir(parents=True)

            frame_columns: dict[str, np.ndarray | list[str]] = {
                "frame_index": np.arange(bin_count, dtype=np.int64),
                "timestamp_ns": frame_timestamps,
                "rgb_path": frames_source["rgb_path"].astype(str).tolist(),
                "human_pose_nearest_error_ms": human_pose_error_ms,
                "gripper_pose_nearest_error_ms": gripper_pose_error_ms,
            }
            expand_pose_columns(
                frame_columns, "human_camera_world", transforms_world_human_camera
            )
            expand_pose_columns(
                frame_columns, "human_camera_common", transforms_common_human_camera
            )
            expand_pose_columns(
                frame_columns, "gripper_device_world", transforms_world_gripper_device
            )
            expand_pose_columns(
                frame_columns, "gripper_device_common", transforms_common_gripper_device
            )
            append_xyz(frame_columns, "tcp_position_world_m", tcp_position_world)
            append_xyz(frame_columns, "tcp_position_common_m", tcp_position_common)
            append_xyz(
                frame_columns, "tcp_velocity_world_m_s", tcp_velocity_world
            )
            append_xyz(
                frame_columns, "tcp_velocity_common_m_s", tcp_velocity_common
            )
            append_xyz(
                frame_columns,
                "tcp_position_current_human_camera_m",
                tcp_position_human_camera,
            )
            append_uv(frame_columns, "tcp_pixel_original", uv_original)
            append_uv(frame_columns, "tcp_pixel_canonical", uv_canonical)
            frame_columns["tcp_in_front_mask"] = in_front.astype(np.uint8)
            frame_columns["tcp_in_image_bounds_mask"] = in_bounds.astype(np.uint8)
            frame_columns["pose_sync_valid_mask"] = pose_sync_valid.astype(np.uint8)
            frame_columns["tcp_visibility_frustum_mask"] = visibility_frustum.astype(
                np.uint8
            )
            frames_output = pd.DataFrame(frame_columns)

            signal_columns: dict[str, np.ndarray] = {
                "signal_index": np.arange(len(signal_timestamps), dtype=np.int64),
                "timestamp_ns": signal_timestamps,
                "time_from_interaction_start_s": (signal_timestamps - start_ns) * 1e-9,
                "gripper_pose_nearest_error_ms": gripper_signal_pose_error_ms,
                "contact_force_gt": contact.astype(np.uint8),
                "force_resultant_n": resultant,
                "power_signed_common_w": power_common,
            }
            append_xyz(
                signal_columns,
                "force_external_on_rig_common_n",
                force_common,
            )
            append_xyz(
                signal_columns,
                "force_on_object_candidate_common_n",
                -force_common,
            )
            append_xyz(
                signal_columns,
                "force_external_on_rig_world_n",
                force_world,
            )
            append_xyz(
                signal_columns,
                "force_on_object_candidate_world_n",
                -force_world,
            )
            append_xyz(signal_columns, "tcp_velocity_common_m_s", velocity_common)
            append_xyz(signal_columns, "tcp_velocity_world_m_s", velocity_world)
            append_xyz(signal_columns, "tcp_position_common_m", tcp_signal_common)
            append_xyz(signal_columns, "tcp_position_world_m", tcp_signal_world)
            signals_output = pd.DataFrame(signal_columns)

            bin_columns: dict[str, np.ndarray] = {
                "frame_index": np.arange(bin_count, dtype=np.int64),
                "frame_timestamp_ns": frame_timestamps,
                "bin_start_ns": edges_ns[:-1],
                "bin_end_ns": edges_ns[1:],
                "bin_duration_s": bin_duration,
                "signal_sample_count": all_sample_count,
                "contact_sample_count": contact_sample_count,
                "contact_occupancy_samples": contact_occupancy_samples,
                "contact_duration_s": contact_duration,
                "contact_occupancy_duration": contact_occupancy_duration,
                "has_contact_interval_mask": (contact_duration > 0.0).astype(np.uint8),
                "peak_force_resultant_n": peak_resultant,
                "peak_force_signal_timestamp_ns": peak_signal_timestamp_ns,
                "signed_energy_external_on_rig_j": signed_energy,
                "signed_energy_on_object_candidate_j": -signed_energy,
                "tcp_path_length_contact_m": path_length_contact,
                "tcp_path_length_all_m": path_length_all,
            }
            append_xyz(
                bin_columns,
                "contact_force_external_on_rig_mean_common_n",
                force_mean_contact,
            )
            append_xyz(
                bin_columns,
                "contact_force_on_object_candidate_mean_common_n",
                -force_mean_contact,
            )
            append_xyz(
                bin_columns,
                "peak_force_external_on_rig_common_n",
                force_at_peak_common,
            )
            append_xyz(
                bin_columns,
                "peak_force_on_object_candidate_common_n",
                -force_at_peak_common,
            )
            append_xyz(
                bin_columns,
                "contact_tcp_velocity_mean_common_m_s",
                velocity_mean_contact,
            )
            bins_output = pd.DataFrame(bin_columns)

            da3_arrays, da3_records, membership = window_arrays(
                frame_timestamps,
                frames_source["rgb_path"].astype(str).tolist(),
                transforms_world_human_camera,
                intrinsics_canonical,
                args.da3_window,
                args.da3_stride,
            )
            frames_output["da3_window_membership_count"] = membership

            frames_output.to_csv(sample_root / "frames.csv", index=False)
            signals_output.to_csv(sample_root / "signals_common.csv", index=False)
            bins_output.to_csv(sample_root / "bins.csv", index=False)
            np.savez_compressed(
                sample_root / "frames_geometry.npz",
                timestamps_ns=frame_timestamps,
                K_original=intrinsics_original,
                K_canonical=intrinsics_canonical,
                original_to_canonical=transform_original_canonical,
                T_world_human_camera=transforms_world_human_camera,
                T_common_human_camera=transforms_common_human_camera,
                T_world_gripper_device=transforms_world_gripper_device,
                T_common_gripper_device=transforms_common_gripper_device,
                T_common_world=transform_common_world,
                tcp_position_world_m=tcp_position_world,
                tcp_position_common_m=tcp_position_common,
                tcp_velocity_world_m_s=tcp_velocity_world,
                tcp_velocity_common_m_s=tcp_velocity_common,
                tcp_position_current_human_camera_m=tcp_position_human_camera,
                tcp_pixel_original_uv=uv_original,
                tcp_pixel_canonical_uv=uv_canonical,
                tcp_visibility_frustum_mask=visibility_frustum.astype(np.uint8),
                reference_frame_index=np.asarray(
                    reference_frame_index, dtype=np.int64
                ),
            )
            np.savez_compressed(
                sample_root / "signals_common.npz",
                timestamps_ns=signal_timestamps,
                force_external_on_rig_common_n=force_common,
                force_on_object_candidate_common_n=-force_common,
                force_external_on_rig_world_n=force_world,
                force_on_object_candidate_world_n=-force_world,
                tcp_velocity_common_m_s=velocity_common,
                tcp_velocity_world_m_s=velocity_world,
                tcp_position_common_m=tcp_signal_common,
                tcp_position_world_m=tcp_signal_world,
                force_resultant_n=resultant,
                contact_force_gt=contact.astype(np.uint8),
                power_signed_common_w=power_common,
            )
            np.savez_compressed(
                sample_root / "bins.npz",
                frame_timestamps_ns=frame_timestamps,
                bin_edges_ns=edges_ns,
                peak_force_resultant_n=peak_resultant,
                peak_force_signal_timestamp_ns=peak_signal_timestamp_ns,
                peak_force_external_on_rig_common_n=force_at_peak_common,
                peak_force_on_object_candidate_common_n=-force_at_peak_common,
                contact_force_external_on_rig_mean_common_n=force_mean_contact,
                contact_force_on_object_candidate_mean_common_n=-force_mean_contact,
                contact_tcp_velocity_mean_common_m_s=velocity_mean_contact,
                signed_energy_external_on_rig_j=signed_energy,
                signed_energy_on_object_candidate_j=-signed_energy,
                contact_duration_s=contact_duration,
                contact_occupancy_duration=contact_occupancy_duration,
                tcp_path_length_contact_m=path_length_contact,
                tcp_path_length_all_m=path_length_all,
            )
            np.savez_compressed(sample_root / "da3_windows_gt.npz", **da3_arrays)
            atomic_json(
                sample_root / "da3_windows.json",
                {
                    "sample_id": sample_id,
                    "window_length": args.da3_window,
                    "stride": args.da3_stride,
                    "reference_strategy": "middle",
                    "reference_local_index": args.da3_window // 2,
                    "canonical_size": [args.canonical_size, args.canonical_size],
                    "coordinate_contract": {
                        "T_window_reference_camera": (
                            "maps each current camera coordinate into the window "
                            "reference-camera coordinate"
                        ),
                        "E_camera_window_reference_w2c": (
                            "maps window reference-camera coordinates into each "
                            "current camera; OpenCV w2c"
                        ),
                    },
                    "metric_scale_target": {
                        "status": "pending_da3_prediction",
                        "why": (
                            "A metric multiplier is a relation between DA3's predicted "
                            "relative reconstruction and the metric SLAM trajectory; it "
                            "cannot be defined from ground truth alone."
                        ),
                        "definition": (
                            "Fit C_gt = s * R_align * C_da3 + t_align by Umeyama "
                            "on window camera centers; save positive s."
                        ),
                        "available_ground_truth_now": [
                            "gt_camera_centers_window_reference_m",
                            "gt_trajectory_rms_radius_m",
                            "gt_max_camera_center_baseline_m",
                            "metric E_w2c and K_canonical",
                        ],
                    },
                    "windows": da3_records,
                },
            )

            sample_record = {
                "version": VERSION,
                "sample_id": sample_id,
                "scene": source_record["scene"],
                "recording": source_record["recording"],
                "object_index": source_record["object_index"],
                "action": source_record["action"],
                "interaction_start_ns": start_ns,
                "interaction_end_ns": end_ns,
                "common_frame": {
                    "definition": (
                        "rectified aria_human RGB camera at the middle RGB frame"
                    ),
                    "reference_frame_index": reference_frame_index,
                    "reference_timestamp_ns": int(
                        frame_timestamps[reference_frame_index]
                    ),
                },
                "counts": {
                    "frames": bin_count,
                    "signals": int(len(signal_timestamps)),
                    "da3_windows": int(len(da3_records)),
                    "da3_gt_scale_observable_windows_0p03m": int(
                        da3_arrays["gt_scale_observable_mask_0p03m"].sum()
                    ),
                    "tcp_frustum_visible_frames": int(visibility_frustum.sum()),
                },
                "paths": {
                    "frames_csv": f"samples/{sample_id}/frames.csv",
                    "frames_npz": f"samples/{sample_id}/frames_geometry.npz",
                    "signals_csv": f"samples/{sample_id}/signals_common.csv",
                    "signals_npz": f"samples/{sample_id}/signals_common.npz",
                    "bins_csv": f"samples/{sample_id}/bins.csv",
                    "bins_npz": f"samples/{sample_id}/bins.npz",
                    "da3_windows_json": f"samples/{sample_id}/da3_windows.json",
                    "da3_windows_npz": f"samples/{sample_id}/da3_windows_gt.npz",
                    "projection_review_video": (
                        f"projection_review/{sample_id}_tcp_projection.mp4"
                    ),
                },
                "invariants": {
                    "max_force_n_from_bins": peak_global,
                    "work_signed_external_on_rig_j_from_bins": energy_global,
                    "work_j_from_bins": abs(energy_global),
                    "max_power_rotation_invariance_error_w": (
                        max_power_invariance_error
                    ),
                },
            }
            atomic_json(sample_root / "manifest.json", sample_record)
            output_records.append(sample_record)
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "frames": bin_count,
                    "signals": len(signal_timestamps),
                    "da3_windows": len(da3_records),
                    "da3_gt_scale_observable_windows_0p03m": int(
                        da3_arrays["gt_scale_observable_mask_0p03m"].sum()
                    ),
                    "tcp_frustum_visible_frames": int(visibility_frustum.sum()),
                    "tcp_frustum_visible_ratio": float(visibility_frustum.mean()),
                    "max_force_n": peak_global,
                    "work_signed_j": energy_global,
                    "work_j": abs(energy_global),
                    "max_power_rotation_invariance_error_w": (
                        max_power_invariance_error
                    ),
                    "max_human_pose_nearest_error_ms": float(
                        human_pose_error_ms.max()
                    ),
                    "max_gripper_pose_nearest_error_ms": float(
                        max(
                            gripper_pose_error_ms.max(),
                            gripper_signal_pose_error_ms.max(),
                        )
                    ),
                }
            )

        with (temporary_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for record in output_records:
                handle.write(json.dumps(record) + "\n")
        pd.DataFrame(summary_rows).to_csv(temporary_root / "summary.csv", index=False)
        contract = {
            "version": VERSION,
            "status": "geometry_generated_da3_scale_pending",
            "transform_notation": "T_A_B maps coordinates from B into A",
            "common_frame": (
                "per interaction: rectified aria_human RGB camera at middle RGB frame"
            ),
            "tcp": {
                "definition": "fixed calibrated tool point",
                "device_to_tcp_offset_device_m": tool_offset_device.tolist(),
                "note": (
                    "This calibrated tool point is not guaranteed to coincide with "
                    "the instantaneous left/right finger surface contact."
                ),
            },
            "human_camera": {
                "rectified_transform_formula": (
                    "T_human_device_rect_camera = "
                    "T_device_camera @ pinhole_T_device_camera"
                ),
                "K_original": intrinsics_original.tolist(),
                "image_size_original": [original_height, original_width],
                "canonical_size": [args.canonical_size, args.canonical_size],
                "original_to_canonical": transform_original_canonical.tolist(),
                "K_canonical": intrinsics_canonical.tolist(),
            },
            "visibility_mask": {
                "name": "tcp_visibility_frustum_mask",
                "definition": (
                    "positive camera z, projected uv within image, and both pose "
                    "streams synchronized within 5 ms"
                ),
                "occlusion_tested": False,
                "warning": (
                    "This is a geometric projection-valid mask, not proof that the "
                    "TCP is visually unoccluded."
                ),
            },
            "force_velocity": {
                "common_frame": True,
                "published_force_semantics": (
                    "force_external_on_rig; unchanged from t2-label-v2"
                ),
                "opposite_candidate": (
                    "force_on_object_candidate = -force_external_on_rig"
                ),
                "warning": (
                    "The hidden official vector sign convention is not released. "
                    "Peak magnitude and final abs(work) are sign invariant; vector "
                    "heads must select an explicit convention."
                ),
                "power_formula": (
                    "force_external_on_rig_common dot tcp_velocity_common"
                ),
                "rotation_invariance_checked": True,
            },
            "rgb_bins": {
                "definition": (
                    "one frame-centered time bin per RGB frame, clipped by the "
                    "manual interaction boundaries"
                ),
                "signed_energy": (
                    "exact piecewise-linear split of contact-only F dot v trapezoids"
                ),
                "global_invariants": [
                    "max(bin_peak_force) == max_force_n",
                    (
                        "sum(bin_signed_energy_external_on_rig) == "
                        "work_signed_j"
                    ),
                    (
                        "abs(sum(bin_signed_energy_external_on_rig)) == "
                        "work_j"
                    ),
                ],
            },
            "da3": {
                "window_length": args.da3_window,
                "stride": args.da3_stride,
                "reference_strategy": "middle",
                "metric_scale_multiplier_status": "pending_da3_prediction",
                "relative_depth_from_existing_camera_depth_used": False,
            },
            "inputs": {
                "dataset_root": str(dataset_root),
                "t2_label_root": str(t2_root),
                "human_trajectory": str(human_trajectory_path),
                "gripper_trajectory": str(gripper_trajectory_path),
                "human_calibration": str(human_calibration_path),
            },
            "source_sha256": {
                "t2_manifest": sha256_file(source_manifest_path),
                "t2_label_config": sha256_file(source_config_path),
                "human_trajectory": sha256_file(human_trajectory_path),
                "gripper_trajectory": sha256_file(gripper_trajectory_path),
                "human_calibration": sha256_file(human_calibration_path),
                "generator": sha256_file(Path(__file__).resolve()),
            },
        }
        atomic_json(temporary_root / "geometry_contract.json", contract)
        (temporary_root / "README.md").write_text(
            README_MARKER
            + "\n\n"
            "Derived, immutable inputs for the first HoI T2 RGB-only model. "
            "Raw HoI data and t2-label-v2 are referenced but never modified.\n\n"
            "Each sample contains frame-rate poses/TCP projections, F/T-rate force "
            "and TCP motion in one common frame, exact RGB-bin targets, and metric "
            "camera trajectory supervision for DA3 sliding windows.\n\n"
            "The DA3 metric scale multiplier remains explicitly pending until a "
            "free-running DA3 prediction exists. Existing camera_depth arrays are "
            "relative Depth Anything V2 predictions and are intentionally ignored.\n",
            encoding="utf-8",
        )
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(f"DONE output={output_root} samples={len(output_records)}")
    print(f"frames={sum(row['frames'] for row in summary_rows)}")
    print(f"signals={sum(row['signals'] for row in summary_rows)}")
    print("da3_metric_scale_target=pending_da3_prediction")


if __name__ == "__main__":
    main()
