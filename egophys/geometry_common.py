#!/usr/bin/env python3
"""Shared geometry and timestamp utilities for HoI T2 geometry-v1.

Transform names follow ``T_A_B``: a homogeneous transform mapping coordinates
from frame B into frame A.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


TRANSLATION_COLUMNS = [
    "tx_world_device",
    "ty_world_device",
    "tz_world_device",
]
QUATERNION_COLUMNS = [
    "qx_world_device",
    "qy_world_device",
    "qz_world_device",
    "qw_world_device",
]
LINEAR_VELOCITY_COLUMNS = [
    "device_linear_velocity_x_device",
    "device_linear_velocity_y_device",
    "device_linear_velocity_z_device",
]
ANGULAR_VELOCITY_COLUMNS = [
    "angular_velocity_x_device",
    "angular_velocity_y_device",
    "angular_velocity_z_device",
]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent
    ) as handle:
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


def nearest_indices(source_timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    insertion = np.searchsorted(source_timestamps, targets, side="left")
    right = np.clip(insertion, 0, len(source_timestamps) - 1)
    left = np.clip(insertion - 1, 0, len(source_timestamps) - 1)
    choose_right = (
        np.abs(source_timestamps[right] - targets)
        < np.abs(source_timestamps[left] - targets)
    )
    return np.where(choose_right, right, left)


def normalize_quaternions(quaternions_xyzw: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions_xyzw, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-norm quaternion")
    return quaternions_xyzw / norms


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path quaternion SLERP.

    Args:
        q0, q1: ``[N,4]`` quaternions in xyzw order.
        alpha: ``[N]`` interpolation fractions.
    """
    q0 = normalize_quaternions(q0)
    q1 = normalize_quaternions(q1)
    dot = np.einsum("ij,ij->i", q0, q1)
    flip = dot < 0.0
    q1 = q1.copy()
    q1[flip] *= -1.0
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    result = np.empty_like(q0)
    nearly_equal = dot > 0.9995
    if np.any(nearly_equal):
        a = alpha[nearly_equal, None]
        result[nearly_equal] = normalize_quaternions(
            (1.0 - a) * q0[nearly_equal] + a * q1[nearly_equal]
        )
    regular = ~nearly_equal
    if np.any(regular):
        theta = np.arccos(dot[regular])
        sin_theta = np.sin(theta)
        a = alpha[regular]
        w0 = np.sin((1.0 - a) * theta) / sin_theta
        w1 = np.sin(a * theta) / sin_theta
        result[regular] = (
            w0[:, None] * q0[regular] + w1[:, None] * q1[regular]
        )
    return normalize_quaternions(result)


def quaternion_xyzw_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions to ``[N,3,3]`` matrices."""
    q = normalize_quaternions(np.asarray(quaternions, dtype=np.float64))
    x, y, z, w = q.T
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrices[:, 0, 1] = 2.0 * (xy - wz)
    matrices[:, 0, 2] = 2.0 * (xz + wy)
    matrices[:, 1, 0] = 2.0 * (xy + wz)
    matrices[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrices[:, 1, 2] = 2.0 * (yz - wx)
    matrices[:, 2, 0] = 2.0 * (xz - wy)
    matrices[:, 2, 1] = 2.0 * (yz + wx)
    matrices[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrices


def matrix_to_quaternion_xyzw(matrices: np.ndarray) -> np.ndarray:
    """Convert ``[N,3,3]`` rotation matrices to canonical xyzw quaternions."""
    matrices = np.asarray(matrices, dtype=np.float64)
    result = np.empty((len(matrices), 4), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = np.sqrt(
                    1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
                ) * 2.0
                w = (matrix[2, 1] - matrix[1, 2]) / scale
                x = 0.25 * scale
                y = (matrix[0, 1] + matrix[1, 0]) / scale
                z = (matrix[0, 2] + matrix[2, 0]) / scale
            elif axis == 1:
                scale = np.sqrt(
                    1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
                ) * 2.0
                w = (matrix[0, 2] - matrix[2, 0]) / scale
                x = (matrix[0, 1] + matrix[1, 0]) / scale
                y = 0.25 * scale
                z = (matrix[1, 2] + matrix[2, 1]) / scale
            else:
                scale = np.sqrt(
                    1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
                ) * 2.0
                w = (matrix[1, 0] - matrix[0, 1]) / scale
                x = (matrix[0, 2] + matrix[2, 0]) / scale
                y = (matrix[1, 2] + matrix[2, 1]) / scale
                z = 0.25 * scale
        result[index] = [x, y, z, w]
    result = normalize_quaternions(result)
    result[result[:, 3] < 0.0] *= -1.0
    return result


def make_transforms(translations: np.ndarray, quaternions_xyzw: np.ndarray) -> np.ndarray:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(translations), axis=0)
    transforms[:, :3, :3] = quaternion_xyzw_to_matrix(quaternions_xyzw)
    transforms[:, :3, 3] = translations
    return transforms


def invert_transforms(transforms: np.ndarray) -> np.ndarray:
    transforms = np.asarray(transforms, dtype=np.float64)
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(transforms), axis=0)
    rotation_t = np.swapaxes(transforms[:, :3, :3], 1, 2)
    result[:, :3, :3] = rotation_t
    result[:, :3, 3] = -np.einsum(
        "nij,nj->ni", rotation_t, transforms[:, :3, 3]
    )
    return result


class PoseTrajectory:
    """Timestamp-indexed SE(3) trajectory with linear translation and SLERP."""

    def __init__(self, table: pd.DataFrame, name: str):
        required = {
            "timestamp",
            *TRANSLATION_COLUMNS,
            *QUATERNION_COLUMNS,
            *LINEAR_VELOCITY_COLUMNS,
            *ANGULAR_VELOCITY_COLUMNS,
        }
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} trajectory missing columns: {sorted(missing)}")
        table = table.sort_values("timestamp")
        if table[list(required)].isna().any().any():
            raise ValueError(f"{name} trajectory has NaN in required columns")
        self.name = name
        self.timestamps = table["timestamp"].to_numpy(dtype=np.int64)
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError(f"{name} timestamps are not strictly increasing")
        self.translations = table[TRANSLATION_COLUMNS].to_numpy(dtype=np.float64)
        self.quaternions = normalize_quaternions(
            table[QUATERNION_COLUMNS].to_numpy(dtype=np.float64)
        )
        self.linear_velocity = table[LINEAR_VELOCITY_COLUMNS].to_numpy(
            dtype=np.float64
        )
        self.angular_velocity = table[ANGULAR_VELOCITY_COLUMNS].to_numpy(
            dtype=np.float64
        )

    def interpolate(
        self, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        targets = np.asarray(targets, dtype=np.int64)
        if targets.min() < self.timestamps[0] or targets.max() > self.timestamps[-1]:
            raise ValueError(f"{self.name} interpolation target outside trajectory")
        right = np.searchsorted(self.timestamps, targets, side="left")
        right = np.clip(right, 1, len(self.timestamps) - 1)
        left = right - 1
        exact_right = self.timestamps[right] == targets
        left[exact_right] = right[exact_right]
        denominator = self.timestamps[right] - self.timestamps[left]
        alpha = np.zeros(len(targets), dtype=np.float64)
        nonzero = denominator != 0
        alpha[nonzero] = (
            (targets[nonzero] - self.timestamps[left[nonzero]])
            / denominator[nonzero]
        )

        translations = (
            (1.0 - alpha[:, None]) * self.translations[left]
            + alpha[:, None] * self.translations[right]
        )
        quaternions = slerp_xyzw(
            self.quaternions[left], self.quaternions[right], alpha
        )
        linear_velocity = (
            (1.0 - alpha[:, None]) * self.linear_velocity[left]
            + alpha[:, None] * self.linear_velocity[right]
        )
        angular_velocity = (
            (1.0 - alpha[:, None]) * self.angular_velocity[left]
            + alpha[:, None] * self.angular_velocity[right]
        )
        nearest = nearest_indices(self.timestamps, targets)
        nearest_error_ms = np.abs(self.timestamps[nearest] - targets) * 1e-6
        transforms = make_transforms(translations, quaternions)
        return transforms, linear_velocity, angular_velocity, nearest_error_ms


def transform_points(transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        np.einsum("nij,nj->ni", transforms[:, :3, :3], points)
        + transforms[:, :3, 3]
    )


def rotate_vectors(transforms: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", transforms[:, :3, :3], vectors)


def expand_pose_columns(
    output: dict[str, np.ndarray],
    prefix: str,
    transforms: np.ndarray,
) -> None:
    translations = transforms[:, :3, 3]
    quaternions = matrix_to_quaternion_xyzw(transforms[:, :3, :3])
    for axis, values in zip("xyz", translations.T):
        output[f"{prefix}_position_{axis}_m"] = values
    for axis, values in zip(("qx", "qy", "qz", "qw"), quaternions.T):
        output[f"{prefix}_{axis}"] = values


def frame_centered_edges(
    frame_timestamps: np.ndarray, interaction_start_ns: int, interaction_end_ns: int
) -> np.ndarray:
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64)
    if np.any(np.diff(frame_timestamps) <= 0):
        raise ValueError("Frame timestamps are not strictly increasing")
    if not (
        interaction_start_ns <= frame_timestamps[0]
        and frame_timestamps[-1] <= interaction_end_ns
    ):
        raise ValueError("RGB frames fall outside interaction")
    middle = (
        frame_timestamps[:-1]
        + (frame_timestamps[1:] - frame_timestamps[:-1]) // 2
    )
    edges = np.concatenate(
        (
            np.asarray([interaction_start_ns], dtype=np.int64),
            middle,
            np.asarray([interaction_end_ns], dtype=np.int64),
        )
    )
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Frame-centered bin edges are not strictly increasing")
    return edges


def sliding_window_starts(length: int, window: int, stride: int) -> list[int]:
    if length < window:
        raise ValueError(f"Sequence length {length} is shorter than window {window}")
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts
