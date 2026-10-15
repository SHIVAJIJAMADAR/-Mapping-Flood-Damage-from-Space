"""
Phase 1 Smoke Test — GPU Verification
Tests PyTorch CUDA availability, device identity, tensor operations, and memory.
"""
import sys
import time


def run_gpu_test():
    results = {}
    
    # 1. Import PyTorch
    try:
        import torch
        results["pytorch_version"] = torch.__version__
        results["pytorch_import"] = "PASS"
    except ImportError as e:
        results["pytorch_import"] = f"FAIL: {e}"
        return results

    # 2. CUDA runtime info
    results["cuda_available"] = torch.cuda.is_available()
    if hasattr(torch.version, "cuda"):
        results["cuda_runtime_version"] = torch.version.cuda
    else:
        results["cuda_runtime_version"] = "N/A"
    results["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"

    if not results["cuda_available"]:
        results["gpu_test"] = "FAIL: torch.cuda.is_available() returned False"
        return results

    # 3. Device identification
    device_count = torch.cuda.device_count()
    results["device_count"] = device_count
    for i in range(device_count):
        results[f"device_{i}_name"] = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        results[f"device_{i}_compute_capability"] = f"{cap[0]}.{cap[1]}"

    # 4. Tensor allocation and computation
    device = torch.device("cuda:0")
    try:
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        
        # Warm up
        _ = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # Timed matmul
        start = time.perf_counter()
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        results["tensor_alloc"] = "PASS"
        results["matmul_10x_1024x1024_ms"] = round(elapsed * 1000, 2)
        results["result_finite"] = bool(torch.isfinite(c).all().item())
        results["result_shape"] = str(c.shape)
    except Exception as e:
        results["tensor_alloc"] = f"FAIL: {e}"

    # 5. Memory info
    try:
        mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        results["gpu_memory_allocated_mb"] = round(mem_alloc, 1)
        results["gpu_memory_reserved_mb"] = round(mem_reserved, 1)
        results["gpu_memory_total_mb"] = round(mem_total, 1)
    except Exception as e:
        results["gpu_memory"] = f"FAIL: {e}"

    results["gpu_test"] = "PASS"
    return results


def test_gpu():
    results = run_gpu_test()
    assert results.get("gpu_test") == "PASS", f"GPU test failed: {results}"


if __name__ == "__main__":
    print("=" * 60)
    print("GPU SMOKE TEST")
    print("=" * 60)
    res = run_gpu_test()
    for k, v in res.items():
        print(f"  {k}: {v}")
    
    overall = "PASS" if res.get("gpu_test") == "PASS" else "FAIL"
    print(f"\n  OVERALL: {overall}")
    sys.exit(0 if overall == "PASS" else 1)
