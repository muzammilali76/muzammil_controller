"""
Horizon-weighted gap guidance, velocity-scheduled steering stabilization,
and Time-to-Collision (TTC) longitudinal speed regulation.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .perception_processor import ProcessedScan


@dataclass
class GuidanceConfig:
    """Tunable parameters for steering control and velocity profiling."""

    max_steer_rad: float = 0.4189
    max_steer_rate_rad_s: float = 3.5
    kp_steer: float = 0.78
    kd_steer: float = 0.08
    kyaw_damping: float = 0.04
    corridor_centering_gain: float = 0.24
    min_gap_width_beams: int = 12
    gap_depth_floor_m: float = 1.10
    v_max_mps: float = 4.35
    v_min_mps: float = 1.50
    v_corner_mps: float = 1.85
    brake_decel_distance_m: float = 4.6
    ttc_critical_s: float = 0.18
    wall_repulsion_dist_m: float = 0.38
    wall_repulsion_gain: float = 0.48


class HorizonGapGuidance:
    """
    Computes smooth Ackermann steering and velocity commands from a
    ProcessedScan and vehicle odometry state.
    """

    def __init__(self, config: GuidanceConfig | None = None) -> None:
        self.cfg = config if config is not None else GuidanceConfig()
        self._prev_target_angle: float = 0.0
        self._prev_steer_cmd: float = 0.0

    def reset(self) -> None:
        """Reset internal filter state."""
        self._prev_target_angle = 0.0
        self._prev_steer_cmd = 0.0

    def compute_command(
        self,
        scan: ProcessedScan,
        linear_speed_mps: float,
        yaw_rate_rad_s: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Compute (steering_angle_rad, speed_mps) from conditioned LiDAR scan
        and current vehicle motion state.
        """
        safe_dt = max(float(dt), 1e-3)

        # 1. Select optimal horizon target angle inside navigable free space
        target_angle = self._select_horizon_angle(scan)

        # 2. Blend corridor centering when driving along straight unobstructed corridors
        if scan.forward_clearance_m > 3.8 and abs(target_angle) < math.radians(15.0):
            target_angle += self.cfg.corridor_centering_gain * scan.corridor_bias_rad

        # 3. Near-wall repulsive barrier correction to prevent apex clipping
        if scan.min_range_m < self.cfg.wall_repulsion_dist_m:
            penetration = (
                self.cfg.wall_repulsion_dist_m - scan.min_range_m
            ) / self.cfg.wall_repulsion_dist_m
            repulse_dir = -1.0 if scan.min_range_angle_rad >= 0.0 else 1.0
            target_angle += (
                repulse_dir * self.cfg.wall_repulsion_gain * penetration
            )

        # 4. PD + yaw-rate damped steering law with slew-rate limiting
        angle_rate = (target_angle - self._prev_target_angle) / safe_dt
        self._prev_target_angle = target_angle

        raw_steer = (
            self.cfg.kp_steer * target_angle
            + self.cfg.kd_steer * angle_rate
            - self.cfg.kyaw_damping * yaw_rate_rad_s
        )
        clipped_steer = float(
            np.clip(raw_steer, -self.cfg.max_steer_rad, self.cfg.max_steer_rad)
        )

        max_delta = self.cfg.max_steer_rate_rad_s * safe_dt
        steer_cmd = float(
            np.clip(
                clipped_steer,
                self._prev_steer_cmd - max_delta,
                self._prev_steer_cmd + max_delta,
            )
        )
        self._prev_steer_cmd = steer_cmd

        # 5. Curvature- and clearance-regulated longitudinal speed profile
        speed_cmd = self._compute_speed(
            scan, steer_cmd, linear_speed_mps
        )

        return steer_cmd, speed_cmd

    def _select_horizon_angle(self, scan: ProcessedScan) -> float:
        """
        Identify contiguous navigable gaps and compute an angularly-regularized,
        depth-weighted target heading.
        """
        masked = scan.masked_ranges
        angles = scan.angles
        max_r = float(np.max(masked))
        if max_r <= 0.15:
            return 0.0

        # Dynamic depth threshold adapts to corridor openness
        adaptive_thresh = max(
            self.cfg.gap_depth_floor_m,
            min(0.50 * max_r, 3.4),
        )
        free_mask = masked >= adaptive_thresh
        if not np.any(free_mask):
            free_mask = masked >= (0.35 * max_r)

        best_slice = self._find_best_contiguous_gap(free_mask, masked, angles)
        gap_ranges = masked[best_slice]
        gap_angles = angles[best_slice]

        if len(gap_ranges) == 0:
            return float(angles[int(np.argmax(masked))])

        peak_r = float(np.max(gap_ranges))
        deep_band = gap_ranges >= (0.80 * peak_r)
        candidate_angles = gap_angles[deep_band]
        candidate_ranges = gap_ranges[deep_band]

        continuity_weight = np.exp(
            -0.55 * np.abs(candidate_angles - self._prev_target_angle)
        )
        forward_weight = np.exp(-0.15 * np.abs(candidate_angles))
        weights = (candidate_ranges ** 2) * continuity_weight * forward_weight
        weight_sum = float(np.sum(weights))

        if weight_sum <= 1e-9:
            return float(np.mean(candidate_angles))
        return float(np.sum(candidate_angles * weights) / weight_sum)

    def _find_best_contiguous_gap(
        self,
        free_mask: np.ndarray,
        masked_ranges: np.ndarray,
        angles: np.ndarray,
    ) -> slice:
        """Return the slice corresponding to the highest-scoring navigable gap."""
        padded = np.concatenate(([False], free_mask, [False]))
        transitions = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)

        best_score = -1.0
        best_slice = slice(0, len(masked_ranges))

        for s, e in zip(starts, ends):
            width = e - s
            if width < self.cfg.min_gap_width_beams and len(starts) > 1:
                continue
            seg_r = masked_ranges[s:e]
            seg_a = angles[s:e]
            mean_depth = float(np.mean(seg_r))
            max_depth = float(np.max(seg_r))
            center_a = float(np.mean(seg_a))
            continuity_bonus = math.exp(
                -0.45 * abs(center_a - self._prev_target_angle)
            )
            score = (
                (width ** 0.60)
                * (0.65 * max_depth + 0.35 * mean_depth)
                * continuity_bonus
            )
            if score > best_score:
                best_score = score
                best_slice = slice(s, e)

        return best_slice

    def _compute_speed(
        self,
        scan: ProcessedScan,
        steer_cmd: float,
        current_speed_mps: float,
    ) -> float:
        """
        Regulate speed smoothly based on steering demand, directional clearance,
        and longitudinal Time-to-Collision (TTC) along the steered path.
        """
        dir_mask = np.abs(scan.angles - steer_cmd) <= math.radians(8.0)
        if np.any(dir_mask):
            steered_clearance = float(np.percentile(scan.raw_ranges[dir_mask], 30))
        else:
            steered_clearance = scan.forward_clearance_m

        proj_speed = max(current_speed_mps, 0.30)
        ttc_steered = steered_clearance / proj_speed
        if (steered_clearance < 0.24 and ttc_steered < self.cfg.ttc_critical_s) or scan.min_range_m < 0.14:
            return 0.0

        steer_ratio = min(abs(steer_cmd) / self.cfg.max_steer_rad, 1.0)
        v_steer = self.cfg.v_min_mps + (
            self.cfg.v_max_mps - self.cfg.v_min_mps
        ) * ((1.0 - steer_ratio) ** 1.65)

        effective_clearance = min(scan.forward_clearance_m * 1.15, steered_clearance)
        if effective_clearance < self.cfg.brake_decel_distance_m:
            clearance_ratio = np.clip(
                (effective_clearance - 0.75)
                / max(self.cfg.brake_decel_distance_m - 0.75, 0.5),
                0.0,
                1.0,
            )
            v_clearance = self.cfg.v_corner_mps + (
                self.cfg.v_max_mps - self.cfg.v_corner_mps
            ) * float(clearance_ratio)
        else:
            v_clearance = self.cfg.v_max_mps

        target_speed = min(v_steer, v_clearance)
        return float(np.clip(target_speed, self.cfg.v_min_mps, self.cfg.v_max_mps))
