"""
Phase 2 Test — Checkpoint Contract & Authoritative Class Mapping
Verifies the Kuro Siwo SNUNet-ECAM input/output contract, target class semantics,
normalization constants, and weight parameters.
"""
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
from backend.models.snunet import (
    SNUNet_ECAM,
    CLASS_NO_WATER,
    CLASS_PERMANENT_WATER,
    CLASS_FLOOD_WATER,
    CLASS_NAMES,
    IMG_MEANS,
    IMG_STDS,
    DEM_MEAN,
    DEM_STD,
)


def check_checkpoint_contract():
    results = {}

    # 1. Authoritative Class Mapping Verification
    # Kuro Siwo official dataset specification:
    # 0 = No water (land/background), 1 = Permanent water, 2 = Flood water
    results["class_0_meaning"] = CLASS_NAMES[CLASS_NO_WATER]
    results["class_1_meaning"] = CLASS_NAMES[CLASS_PERMANENT_WATER]
    results["class_2_meaning"] = CLASS_NAMES[CLASS_FLOOD_WATER]

    assert CLASS_NO_WATER == 0, "Class 0 must be No water"
    assert CLASS_PERMANENT_WATER == 1, "Class 1 must be Permanent water"
    assert CLASS_FLOOD_WATER == 2, "Class 2 must be Flood water"
    results["class_contract"] = "PASS"

    # 2. Normalization Constants Verification
    assert len(IMG_MEANS) == 2 and len(IMG_STDS) == 2
    assert abs(IMG_MEANS[0] - 0.0953) < 1e-4, "VV mean mismatch"
    assert abs(IMG_MEANS[1] - 0.0264) < 1e-4, "VH mean mismatch"
    assert abs(DEM_MEAN - 67.0293) < 1e-4, "DEM mean mismatch"
    results["normalization_constants"] = "PASS"

    # 3. Model Architecture & Weights Verification
    ckpt_path = WORKSPACE_ROOT / "checkpoints" / "best_segmentation.pt"
    assert ckpt_path.is_file(), f"Checkpoint missing at {ckpt_path}"

    model = SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    load_result = model.load_state_dict(state_dict, strict=True)
    assert len(load_result.missing_keys) == 0, f"Missing keys: {load_result.missing_keys}"
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    results["weights_strict_load"] = "PASS"

    # 4. Input & Output Tensor Shapes
    model.eval()
    B, C, H, W = 1, 3, 224, 224
    xA = torch.randn(B, C, H, W)
    xB = torch.randn(B, C, H, W)
    with torch.no_grad():
        out = model(xA, xB)

    assert out.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Model output contains NaN or Inf"
    results["tensor_contract_224x224"] = "PASS"

    # 5. Output Class Space
    classes = torch.unique(torch.argmax(out, dim=1)).tolist()
    assert all(c in [0, 1, 2] for c in classes), f"Invalid predicted classes: {classes}"
    results["output_class_space"] = "PASS"

    results["contract_test"] = "PASS"
    return results


def test_checkpoint_contract():
    res = check_checkpoint_contract()
    assert res.get("contract_test") == "PASS", f"Contract test failed: {res}"


if __name__ == "__main__":
    print("=" * 60)
    print("CHECKPOINT CONTRACT VERIFICATION")
    print("=" * 60)
    res = check_checkpoint_contract()
    for k, v in res.items():
        print(f"  {k}: {v}")
    overall = "PASS" if res.get("contract_test") == "PASS" else "FAIL"
    print(f"\n  OVERALL: {overall}")
    sys.exit(0 if overall == "PASS" else 1)
