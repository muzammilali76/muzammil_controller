"""
LiDAR scan conditioning, radial disparity inflation, and corridor geometry
estimation for F1TENTH autonomous racing.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ScanConfig:
    """Configuration parameters for LiDAR preprocessing and obstacle inflation."""

    fov_half_angle_rad: float = math.radians(100.0)
    min_range_m: float = 0.08
    max_horizon_m: float = 11.5
    smoothing_window: int = 5
    disparity_threshold_m: float = 0.35
    car_half_width_m: float = 0.165
    lateral_safety_margin_m: float = 0.115
    bubble_radius_m: float = 0.25


@dataclass
class ProcessedScan:
    """Conditioned forward-hemisphere scan and auxiliary clearance metrics."""

    angles: np.ndarray
    raw_ranges: np.ndarray
    masked_ranges: np.ndarray
    forward_clearance_m: float
    min_range_m: float
    min_range_angle_rad: float
    left_wall_dist_m: float
    right_wall_dist_m: float
    corridor_bias_rad: float


class LidarPerceptionProcessor:
    """
    Conditions raw 2D LaserScan measurements into a disparity-inflated navigable
    horizon profile and estimates lateral corridor symmetry.
    """

    def __init__(self, config: ScanConfig | None = None) -> None:
        self.cfg = config if config is not None else ScanConfig()
        kernel_size = max(1, int(self.cfg.smoothing_window))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self._kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)

    def process(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> ProcessedScan:
        """
        Crop scan to forward field of view, sanitize invalid returns, inflate
        radial disparities and nearest obstacle bubble, and compute wall metrics.
        """
        num_beams = len(ranges)
        if num_beams == 0 or angle_increment <= 0.0:
            raise ValueError("Invalid LaserScan array or non-positive angle_increment.")

        all_angles = angle_min + np.arange(num_beams, dtype=np.float64) * angle_increment

        # Extract lateral wall distances around +-65..95 deg before cropping
        left_wall = self._sector_distance(
            ranges, all_angles, math.radians(65.0), math.radians(95.0)
        )
        right_wall = self._sector_distance(
            ranges, all_angles, math.radians(-95.0), math.radians(-65.0)
        )
        left_diag = self._sector_distance(
            ranges, all_angles, math.radians(30.0), math.radians(55.0)
        )
        right_diag = self._sector_distance(
            ranges, all_angles, math.radians(-55.0), math.radians(-30.0)
        )

        fov_mask = np.abs(all_angles) <= self.cfg.fov_half_angle_rad
        fov_angles = all_angles[fov_mask]
        fov_raw = np.asarray(ranges[fov_mask], dtype=np.float64)

        # Replace NaNs/Infs and clip to horizon bounds
        sanitized = np.where(
            np.isfinite(fov_raw), fov_raw, self.cfg.max_horizon_m
        )
        sanitized = np.clip(
            sanitized, self.cfg.min_range_m, self.cfg.max_horizon_m
        )

        if len(sanitized) >= len(self._kernel):
            smoothed = np.convolve(sanitized, self._kernel, mode="same")
        else:
            smoothed = sanitized.copy()

        # Global nearest obstacle in FOV
        min_idx = int(np.argmin(smoothed))
        min_dist = float(smoothed[min_idx])
        min_angle = float(fov_angles[min_idx])

        # Forward cone clearance within +-10 deg
        fwd_mask = np.abs(fov_angles) <= math.radians(10.0)
        fwd_clearance = (
            float(np.percentile(smoothed[fwd_mask], 35))
            if np.any(fwd_mask)
            else min_dist
        )

        # Apply radial disparity extension and closest-point safety bubble
        masked = self._inflate_disparities(smoothed, angle_increment)
        masked = self._apply_safety_bubble(
            masked, min_idx, min_dist, angle_increment
        )

        # Compute symmetric corridor centering bias
        wall_sum = left_wall + right_wall
        if 0.8 <= wall_sum <= 3.6:
            lateral_error = (left_wall - right_wall) / max(wall_sum, 1.0)
            diag_error = (left_diag - right_diag) / max(left_diag + right_diag, 1.0)
            corridor_bias = 0.35 * lateral_error + 0.25 * diag_error
        else:
            corridor_bias = 0.0

        return ProcessedScan(
            angles=fov_angles,
            raw_ranges=smoothed,
            masked_ranges=masked,
            forward_clearance_m=fwd_clearance,
            min_range_m=min_dist,
            min_range_angle_rad=min_angle,
            left_wall_dist_m=left_wall,
            right_wall_dist_m=right_wall,
            corridor_bias_rad=float(corridor_bias),
        )

    def _sector_distance(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        low_rad: float,
        high_rad: float,
    ) -> float:
        mask = (angles >= low_rad) & (angles <= high_rad)
        if not np.any(mask):
            return self.cfg.max_horizon_m
        sector = np.asarray(ranges[mask], dtype=np.float64)
        sector = np.where(np.isfinite(sector), sector, self.cfg.max_horizon_m)
        sector = np.clip(sector, self.cfg.min_range_m, self.cfg.max_horizon_m)
        return float(np.median(sector))

    def _inflate_disparities(
        self, smoothed: np.ndarray, angle_inc: float
    ) -> np.ndarray:
        """
        Extend the nearer range across radial step discontinuities so the car
        clears wall corners and track obstacles with full half-width margin.
        """
        masked = smoothed.copy()
        diffs = np.diff(smoothed)
        jump_indices = np.flatnonzero(
            np.abs(diffs) > self.cfg.disparity_threshold_m
        )
        effective_radius = (
            self.cfg.car_half_width_m + self.cfg.lateral_safety_margin_m
        )
        n = len(smoothed)

        for idx in jump_indices:
            left_val = smoothed[idx]
            right_val = smoothed[idx + 1]
            if left_val < right_val:
                near_dist = max(left_val, self.cfg.min_range_m)
                half_span = min(
                    math.atan2(effective_radius, near_dist), math.radians(28.0)
                )
                num_samples = int(math.ceil(half_span / angle_inc))
                end_i = min(n, idx + 1 + num_samples)
                masked[idx + 1 : end_i] = np.minimum(
                    masked[idx + 1 : end_i], near_dist
                )
            else:
                near_dist = max(right_val, self.cfg.min_range_m)
                half_span = min(
                    math.atan2(effective_radius, near_dist), math.radians(28.0)
                )
                num_samples = int(math.ceil(half_span / angle_inc))
                start_i = max(0, idx + 1 - num_samples)
                masked[start_i : idx + 1] = np.minimum(
                    masked[start_i : idx + 1], near_dist
                )

        return masked

    def _apply_safety_bubble(
        self,
        masked: np.ndarray,
        center_idx: int,
        min_dist: float,
        angle_inc: float,
    ) -> np.ndarray:
        """Zero out rays within the physical collision bubble of the closest point."""
        out = masked.copy()
        safe_d = max(min_dist, self.cfg.min_range_m)
        bubble_angle = min(
            math.atan2(self.cfg.bubble_radius_m, safe_d), math.radians(22.0)
        )
        span = int(math.ceil(bubble_angle / angle_inc))
        lo = max(0, center_idx - span)
        hi = min(len(out), center_idx + span + 1)
        out[lo:hi] = 0.0
        return out
