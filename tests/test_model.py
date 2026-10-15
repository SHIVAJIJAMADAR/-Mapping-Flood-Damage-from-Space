"""
Phase 1 Smoke Test — Model Verification
Loads SNUNet-ECAM checkpoint, validates weights, and benchmarks forward pass on GPU/CPU.
"""
import os
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def run_model_test():
    results = {}

    # 1. Import torch and model architecture
    try:
        import torch
        from backend.models.snunet import SNUNet_ECAM
        results["torch_import"] = "PASS"
        results["snunet_import"] = "PASS"
    except ImportError as e:
        results["import_error"] = str(e)
        return results

    # 2. Check checkpoint existence
    workspace_root = Path(__file__).resolve().parent.parent
    ckpt_path = workspace_root / "checkpoints" / "best_segmentation.pt"
    if not ckpt_path.is_file():
        # Fallback to scratch artifact if not in checkpoints
        alt_path = Path(r"C:\Users\nikit\.gemini\antigravity-ide\brain\2a29bb99-1ea7-487a-a08b-787cc52e20e4\scratch\best_segmentation.pt")
        if alt_path.is_file():
            ckpt_path = alt_path
        else:
            results["checkpoint"] = f"FAIL: Not found at {ckpt_path}"
            return results

    results["checkpoint_path"] = str(ckpt_path)
    results["checkpoint_size_mb"] = round(os.path.getsize(ckpt_path) / (1024 ** 2), 2)

    # 3. Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results["device"] = str(device)
    if device.type == "cuda":
        results["device_name"] = torch.cuda.get_device_name(0)

    # 4. Instantiate model
    try:
        model = SNUNet_ECAM(in_channels=3, out_ch=3, base_channel=32)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results["total_parameters"] = total_params
        results["trainable_parameters"] = trainable_params
    except Exception as e:
        results["instantiation"] = f"FAIL: {e}"
        return results

    # 5. Load weights
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        results["weights_load"] = "PASS"
        if "epoch" in ckpt:
            results["checkpoint_epoch"] = ckpt.get("epoch")
        if "best_loss" in ckpt:
            results["checkpoint_best_loss"] = ckpt.get("best_loss")
    except Exception as e:
        results["weights_load"] = f"FAIL: {e}"
        return results

    # 6. Forward pass with synthetic SAR dual-temporal tensor (B=1, C=3, H=256, W=256)
    try:
        xA = torch.randn(1, 3, 256, 256, device=device)
        xB = torch.randn(1, 3, 256, 256, device=device)

        # Warm up
        with torch.no_grad():
            _ = model(xA, xB)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed inference
        start = time.perf_counter()
        iterations = 5
        with torch.no_grad():
            for _ in range(iterations):
                out = model(xA, xB)
            if device.type == "cuda":
                torch.cuda.synchronize()
        latency_ms = round(((time.perf_counter() - start) / iterations) * 1000, 2)

        results["output_shape"] = list(out.shape)
        results["inference_latency_ms"] = latency_ms
        results["output_all_finite"] = bool(torch.isfinite(out).all().item())

        # Validate softmax probabilities
        probs = torch.softmax(out, dim=1)
        prob_sum = probs.sum(dim=1)
        results["prob_sum_is_one"] = bool(torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5))

        # Check unique predicted classes
        pred_classes = torch.unique(torch.argmax(out, dim=1)).cpu().tolist()
        results["predicted_classes_sample"] = pred_classes

        # Memory usage
        if device.type == "cuda":
            results["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)

        results["model_test"] = "PASS"
    except Exception as e:
        results["forward_pass"] = f"FAIL: {e}"
        results["model_test"] = "FAIL"

    return results


def test_model():
    res = run_model_test()
    assert res.get("model_test") == "PASS", f"Model test failed: {res}"


if __name__ == "__main__":
    print("=" * 60)
    print("SNUNET-ECAM MODEL SMOKE TEST")
    print("=" * 60)
    res = run_model_test()
    for k, v in res.items():
        print(f"  {k}: {v}")
    overall = "PASS" if res.get("model_test") == "PASS" else "FAIL"
    print(f"\n  OVERALL: {overall}")
    sys.exit(0 if overall == "PASS" else 1)
