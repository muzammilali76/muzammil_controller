"""Unit and closed-loop tests for the muzammil_controller perception and guidance pipeline."""

import math
import unittest
import numpy as np

from muzammil_controller.horizon_guidance import GuidanceConfig, HorizonGapGuidance
from muzammil_controller.perception_processor import (
    LidarPerceptionProcessor,
    ScanConfig,
)


def _make_synthetic_scan(
    num_beams: int = 1080,
    default_range: float = 8.0,
) -> tuple[np.ndarray, float, float]:
    angle_min = -2.35619449615
    angle_max = 2.35619449615
    angle_inc = (angle_max - angle_min) / (num_beams - 1)
    ranges = np.full(num_beams, default_range, dtype=np.float64)
    return ranges, angle_min, angle_inc


class TestControllerPipeline(unittest.TestCase):
    def test_nan_and_inf_sanitization(self) -> None:
        ranges, angle_min, angle_inc = _make_synthetic_scan()
        ranges[500:520] = np.nan
        ranges[540:560] = np.inf

        processor = LidarPerceptionProcessor(ScanConfig())
        processed = processor.process(ranges, angle_min, angle_inc)

        self.assertTrue(np.all(np.isfinite(processed.raw_ranges)))
        self.assertTrue(np.all(np.isfinite(processed.masked_ranges)))
        self.assertGreater(processed.forward_clearance_m, 4.0)

    def test_obstacle_avoidance_steering_direction(self) -> None:
        ranges, angle_min, angle_inc = _make_synthetic_scan(default_range=7.5)
        angles = angle_min + np.arange(len(ranges)) * angle_inc

        obs_mask = (angles >= math.radians(-22.0)) & (angles <= math.radians(2.0))
        ranges[obs_mask] = 1.25

        processor = LidarPerceptionProcessor(ScanConfig())
        guidance = HorizonGapGuidance(GuidanceConfig())

        processed = processor.process(ranges, angle_min, angle_inc)
        steer_rad, speed_mps = guidance.compute_command(processed, 2.5, 0.0, 0.025)

        self.assertGreater(steer_rad, 0.05)
        self.assertGreater(speed_mps, 0.0)
        self.assertLessEqual(speed_mps, guidance.cfg.v_max_mps)

    def test_emergency_ttc_stop(self) -> None:
        ranges, angle_min, angle_inc = _make_synthetic_scan(default_range=0.12)
        processor = LidarPerceptionProcessor(ScanConfig())
        guidance = HorizonGapGuidance(GuidanceConfig())

        processed = processor.process(ranges, angle_min, angle_inc)
        _, speed_mps = guidance.compute_command(processed, 3.0, 0.0, 0.025)

        self.assertEqual(speed_mps, 0.0)


if __name__ == "__main__":
    unittest.main()
