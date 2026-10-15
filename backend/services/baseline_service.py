"""
Independent Numerical Change-Detection Baseline Service
Computes radar backscatter difference in decibels (delta_VV_dB) and applies
configurable classical thresholds and slope filtering independently of the neural network.
"""

from typing import Dict, Any, Optional
import numpy as np


class BaselineConfig:
    def __init__(
        self,
        db_drop_threshold: float = -3.0,
        max_slope_degrees: float = 15.0,
        min_linear_power: float = 1e-5,
    ):
        self.db_drop_threshold = db_drop_threshold
        self.max_slope_degrees = max_slope_degrees
        self.min_linear_power = min_linear_power


def compute_radar_change_baseline(
    pre_vv_linear: np.ndarray,
    post_vv_linear: np.ndarray,
    slope_deg: np.ndarray,
    valid_mask: np.ndarray,
    config: Optional[BaselineConfig] = None,
) -> Dict[str, Any]:
    """
    Computes decibel conversion, backscatter difference, and classical candidate-change mask.

    Formulas:
      VV_dB = 10 * log10(max(VV_linear, 1e-5))
      delta_VV_dB = VV_post_dB - VV_pre_dB
      raw_change_mask = delta_VV_dB < db_drop_threshold
      slope_filtered_mask = raw_change_mask & (slope_deg < max_slope_degrees)
    """
    if config is None:
        config = BaselineConfig()

    # Mask invalid or negative pixels
    safe_valid = valid_mask & np.isfinite(pre_vv_linear) & np.isfinite(post_vv_linear) & (pre_vv_linear > 0) & (post_vv_linear > 0)

    # Convert linear power to decibels (dB)
    pre_vv_safe = np.where(safe_valid, pre_vv_linear, config.min_linear_power)
    post_vv_safe = np.where(safe_valid, post_vv_linear, config.min_linear_power)

    pre_vv_db = 10.0 * np.log10(np.clip(pre_vv_safe, config.min_linear_power, None))
    post_vv_db = 10.0 * np.log10(np.clip(post_vv_safe, config.min_linear_power, None))

    delta_vv_db = post_vv_db - pre_vv_db
    delta_vv_db[~safe_valid] = 0.0

    # 1. Classical radar drop mask (significant backscatter drop indicates surface smoothing)
    raw_change_mask = (delta_vv_db < config.db_drop_threshold) & safe_valid

    # 2. Slope-filtered candidate mask (water does not accumulate on steep mountain slopes > 15 deg)
    slope_valid = np.isfinite(slope_deg)
    slope_filtered_mask = raw_change_mask & (slope_deg < config.max_slope_degrees) & slope_valid

    # Summary statistics over valid pixels
    valid_deltas = delta_vv_db[safe_valid]
    stats = {
        "mean_delta_db": float(np.mean(valid_deltas)) if len(valid_deltas) > 0 else 0.0,
        "std_delta_db": float(np.std(valid_deltas)) if len(valid_deltas) > 0 else 0.0,
        "min_delta_db": float(np.min(valid_deltas)) if len(valid_deltas) > 0 else 0.0,
        "max_delta_db": float(np.max(valid_deltas)) if len(valid_deltas) > 0 else 0.0,
        "raw_candidate_pixels": int(np.sum(raw_change_mask)),
        "slope_filtered_pixels": int(np.sum(slope_filtered_mask)),
        "threshold_db": config.db_drop_threshold,
        "max_slope_degrees": config.max_slope_degrees,
    }

    return {
        "pre_vv_db": pre_vv_db.astype(np.float32),
        "post_vv_db": post_vv_db.astype(np.float32),
        "delta_vv_db": delta_vv_db.astype(np.float32),
        "raw_change_mask": raw_change_mask.astype(np.uint8),
        "candidate_flood_mask": slope_filtered_mask.astype(np.uint8),
        "stats": stats,
    }
