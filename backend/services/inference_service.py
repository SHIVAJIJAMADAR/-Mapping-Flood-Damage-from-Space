"""
Tiled Sliding-Window Inference Service for SNUNet-ECAM
Performs seamless, windowed dual-temporal SAR flood segmentation with Gaussian/Hann blending.
"""

import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import rasterio

from backend.models.snunet import (
    SNUNet_ECAM,
    CLASS_NO_WATER,
    CLASS_PERMANENT_WATER,
    CLASS_FLOOD_WATER,
    IMG_MEANS,
    IMG_STDS,
    DEM_MEAN,
    DEM_STD,
    SLOPE_MEAN,
    SLOPE_STD,
)


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[str] = None,
        patch_size: int = 224,
        stride: int = 168,
        batch_size: int = 4,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Instantiate and load model
        self.model = SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 2. Build 2D Hann blending window for seamless edge tapering
        wy = np.hanning(patch_size)
        wx = np.hanning(patch_size)
        weight_2d = np.outer(wy, wx).astype(np.float32)
        self.weight_window = np.clip(weight_2d, 1e-4, 1.0)
        self.weight_tensor = torch.from_numpy(self.weight_window).to(self.device)

    def prepare_input_tensors(
        self,
        pre_vv: np.ndarray,
        pre_vh: np.ndarray,
        post_vv: np.ndarray,
        post_vh: np.ndarray,
        dem: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalizes linear SAR backscatter and DEM according to Kuro Siwo contract.
        Returns (pre_3ch, post_3ch) arrays of shape (3, H, W).
        """
        # Linear clamp to prevent negative or infinite backscatter
        pre_vv_clean = np.clip(pre_vv, 1e-5, 2.0)
        pre_vh_clean = np.clip(pre_vh, 1e-5, 1.0)
        post_vv_clean = np.clip(post_vv, 1e-5, 2.0)
        post_vh_clean = np.clip(post_vh, 1e-5, 1.0)

        # Normalize SAR backscatter (Linear power)
        norm_pre_vv = (pre_vv_clean - IMG_MEANS[0]) / IMG_STDS[0]
        norm_pre_vh = (pre_vh_clean - IMG_MEANS[1]) / IMG_STDS[1]
        norm_post_vv = (post_vv_clean - IMG_MEANS[0]) / IMG_STDS[0]
        norm_post_vh = (post_vh_clean - IMG_MEANS[1]) / IMG_STDS[1]

        # Normalize DEM
        norm_dem = (dem - DEM_MEAN) / DEM_STD

        pre_3ch = np.stack([norm_pre_vv, norm_pre_vh, norm_dem], axis=0).astype(np.float32)
        post_3ch = np.stack([norm_post_vv, norm_post_vh, norm_dem], axis=0).astype(np.float32)
        return pre_3ch, post_3ch

    @torch.inference_mode()
    def predict_tiled(
        self,
        pre_3ch: np.ndarray,
        post_3ch: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Executes sliding-window tiled inference across full raster dimensions.

        Returns:
          - probs: (3, H, W) float32 array
          - flood_prob: (H, W) float32 array (class 2)
          - pred_class: (H, W) uint8 array (0, 1, 2)
        """
        _, H, W = pre_3ch.shape
        P = self.patch_size
        S = self.stride

        # Pad raster to ensure complete coverage by sliding window
        pad_y = max(0, P - H) if H < P else (S - ((H - P) % S)) % S
        pad_x = max(0, P - W) if W < P else (S - ((W - P) % S)) % S

        if pad_y > 0 or pad_x > 0:
            pre_padded = np.pad(pre_3ch, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")
            post_padded = np.pad(post_3ch, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")
        else:
            pre_padded = pre_3ch
            post_padded = post_3ch

        pH, pW = pre_padded.shape[1], pre_padded.shape[2]

        # Generate window coordinates
        y_steps = list(range(0, pH - P + 1, S))
        if y_steps[-1] + P < pH:
            y_steps.append(pH - P)

        x_steps = list(range(0, pW - P + 1, S))
        if x_steps[-1] + P < pW:
            x_steps.append(pW - P)

        # Accumulation buffers
        accum_logits = np.zeros((3, pH, pW), dtype=np.float32)
        accum_weights = np.zeros((pH, pW), dtype=np.float32)

        # Build batches
        batch_coords = []
        batch_pre = []
        batch_post = []

        for y in y_steps:
            for x in x_steps:
                batch_coords.append((y, x))
                batch_pre.append(pre_padded[:, y:y+P, x:x+P])
                batch_post.append(post_padded[:, y:y+P, x:x+P])

                if len(batch_pre) == self.batch_size:
                    self._process_batch(
                        batch_pre, batch_post, batch_coords, accum_logits, accum_weights, P
                    )
                    batch_coords, batch_pre, batch_post = [], [], []

        if batch_pre:
            self._process_batch(
                batch_pre, batch_post, batch_coords, accum_logits, accum_weights, P
            )

        # Normalize accumulated logits by weights
        accum_weights = np.maximum(accum_weights, 1e-5)
        stitched_logits = accum_logits[:, :H, :W] / accum_weights[:H, :W]

        # Softmax probabilities
        tensor_logits = torch.from_numpy(stitched_logits)
        probs_tensor = torch.softmax(tensor_logits, dim=0)
        probs = probs_tensor.cpu().numpy()

        # Class predictions (Argmax: 0=no water, 1=permanent water, 2=flood water)
        pred_class = np.argmax(probs, axis=0).astype(np.uint8)
        flood_prob = probs[CLASS_FLOOD_WATER].astype(np.float32)

        # Apply valid mask
        flood_prob[~valid_mask] = 0.0
        pred_class[~valid_mask] = CLASS_NO_WATER

        return {
            "probs": probs,
            "flood_prob": flood_prob,
            "pred_class": pred_class,
        }

    def _process_batch(
        self,
        batch_pre: list,
        batch_post: list,
        batch_coords: list,
        accum_logits: np.ndarray,
        accum_weights: np.ndarray,
        P: int,
    ):
        t_pre = torch.from_numpy(np.stack(batch_pre)).to(self.device)
        t_post = torch.from_numpy(np.stack(batch_post)).to(self.device)

        logits = self.model(t_pre, t_post)  # [B, 3, P, P]
        logits_np = logits.cpu().numpy()

        for i, (y, x) in enumerate(batch_coords):
            accum_logits[:, y:y+P, x:x+P] += logits_np[i] * self.weight_window
            accum_weights[y:y+P, x:x+P] += self.weight_window
