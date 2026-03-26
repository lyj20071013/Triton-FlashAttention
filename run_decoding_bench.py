import torch
import triton
import triton.language as tl
from timeit import Timer
from decoding.flash_decoding_v1 import call_flash_decoding

def pytorch_native_decoding(q, k, v):
    # q: [B, H, 1, D], k: [B, H, L, D]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

def benchmark_decoding():
    torch.manual_seed(42)
    print("=" * 50)
    print("🚀Flash Decoding Profiler")
    print("=" * 50)
    
    # Test extreme context: Q=1, massive KV cache
    configs = [
        {'B': 1, 'H': 16, 'L': 8192, 'D': 64},    # Medium KV Cache
        {'B': 1, 'H': 16, 'L': 65536, 'D': 64},   # Extreme Long-Context KV Cache (64K)
    ]

    for cfg in configs:
        B, H, L, D = cfg['B'], cfg['H'], cfg['L'], cfg['D']
        print(f"\n[Config] Batch: {B}, Heads: {H}, SeqLen: {L}, HeadDim: {D}")
        
        # FP32 test data for Pascal GPU
        q = torch.randn((B, H, 1, D), device='cuda', dtype=torch.float32)
        k = torch.randn((B, H, L, D), device='cuda', dtype=torch.float32)
        v = torch.randn((B, H, L, D), device='cuda', dtype=torch.float32)

        # Correctness verification
        out_torch = pytorch_native_decoding(q, k, v)
        out_triton = call_flash_decoding(q, k, v)
        max_diff = (out_torch - out_triton).abs().max().item()
        print(f"  ├─ Correctness Check (Max Diff): {max_diff:.6f} " + ("✅" if max_diff < 1e-3 else "❌"))

        def run_torch(): pytorch_native_decoding(q, k, v)
        def run_triton(): call_flash_decoding(q, k, v)

        # Warmup
        for _ in range(10): 
            run_torch()
            run_triton()
        torch.cuda.synchronize()

        # Benchmark (1000 iterations for stable microsecond timing)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        results = {}
        for name, fn in [("PyTorch Native", run_torch), ("Flash Decoding (Split-K)", run_triton)]:
            times = []
            for _ in range(1000):
                start_event.record()
                fn()
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            
            # Trim outliers
            times = sorted(times)[100:-100] 
            avg_time = sum(times) / len(times)
            results[name] = avg_time

        # Analyze Memory Bandwidth
        # Decoding I/O: Read Q (negligible), Read K/V (dominant), Write O (negligible)
        # Primary HBM I/O = 2 * (B * H * L * D) * 4 Bytes (FP32)
        total_bytes = 2 * B * H * L * D * 4 
        
        for name, t_ms in results.items():
            bw_gbps = (total_bytes / 1e9) / (t_ms / 1000.0)
            print(f"  ├─ [{name}] Time: {t_ms:.3f} ms | Bandwidth: {bw_gbps:.2f} GB/s")

if __name__ == "__main__":
    benchmark_decoding()