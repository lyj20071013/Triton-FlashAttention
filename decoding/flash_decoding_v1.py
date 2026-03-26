import torch
import triton
import triton.language as tl

@triton.jit
def flash_decoding_stage1(
    Q, K, V, Mid_O, Mid_LSE,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_mob, stride_moh, stride_moc, stride_mod,
    stride_mlb, stride_mlh, stride_mlc,
    B, H, L, D: tl.constexpr,
    BLOCK_KV: tl.constexpr
):
    # 2D Grid: [batch * num_heads, num_chunks]
    pid_bh = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    # 解析当前处理的 batch 和 head
    b = pid_bh // H
    h = pid_bh % H

    # 当前 Chunk 在 KV 序列中的物理偏移
    start_n = pid_chunk * BLOCK_KV
    offs_n = start_n + tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, D)

    # 1. 加载 Query [1, D]
    # Q 的形状是 [B, H, 1, D]，没有序列维度
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)

    # 2. 加载 K 和 V 的局部 Chunk [BLOCK_KV, D]
    k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k_mask = (offs_n[:, None] < L) & (offs_d[None, :] < D)
    k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

    # 3. 计算局部 QK^T
    s = tl.sum(q[None, :] * k, axis=1) # 沿着特征维度 D 规约，得到 [BLOCK_KV]
    s = s * (1.0 / tl.sqrt(tl.cast(D, tl.float32)))

    # Mask 掉超出序列实际长度的部分
    s = tl.where(offs_n < L, s, float('-inf'))

    # 4. 局部 Softmax
    m_i = tl.max(s, axis=0)
    p = tl.exp(s - m_i)
    l_i = tl.sum(p, axis=0)
    
    p_norm = p / l_i
    
    # 局部输出加权 [D]
    o_i = tl.sum(p_norm[:, None] * v, axis=0)
    
    # 局部 LogSumExp
    lse_i = m_i + tl.log(l_i)

    # 5. 写回中间状态 (SRAM -> HBM)
    mo_ptrs = Mid_O + b * stride_mob + h * stride_moh + pid_chunk * stride_moc + offs_d * stride_mod
    tl.store(mo_ptrs, o_i, mask=offs_d < D)

    ml_ptrs = Mid_LSE + b * stride_mlb + h * stride_mlh + pid_chunk * stride_mlc
    tl.store(ml_ptrs, lse_i)

@triton.jit
def flash_decoding_stage2(
    Mid_O, Mid_LSE, Out,
    stride_mob, stride_moh, stride_moc, stride_mod,
    stride_mlb, stride_mlh, stride_mlc,
    stride_ob, stride_oh, stride_od,
    B, H, num_chunks, D: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr
):
    # 1D Grid: [batch * num_heads]
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H

    offs_c = tl.arange(0, BLOCK_CHUNKS)
    offs_d = tl.arange(0, D)

    # 1. 捞回所有局部 LSE [num_chunks]
    ml_ptrs = Mid_LSE + b * stride_mlb + h * stride_mlh + offs_c * stride_mlc
    lse = tl.load(ml_ptrs, mask=offs_c < num_chunks, other=float('-inf'))

    # 2. 二段式归约：计算 Global Max 和 Global Sum
    global_max = tl.max(lse, axis=0)
    exp_lse = tl.exp(lse - global_max)
    global_sum = tl.sum(exp_lse, axis=0)
    
    # 计算每个 chunk 对全局的权重贡献
    weights = exp_lse / global_sum 

    # 3. 捞回所有局部的 O 并进行全局加权合并
    mo_ptrs = Mid_O + b * stride_mob + h * stride_moh + offs_c[:, None] * stride_moc + offs_d[None, :] * stride_mod
    mo_mask = (offs_c[:, None] < num_chunks) & (offs_d[None, :] < D)
    mid_o = tl.load(mo_ptrs, mask=mo_mask, other=0.0)

    final_o = tl.sum(mid_o * weights[:, None], axis=0)

    # 4. 写回最终的 Decoding 结果
    o_ptrs = Out + b * stride_ob + h * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, final_o, mask=offs_d < D)


def call_flash_decoding(q, k, v):
    # 为 1080Ti 强制转换 FP32，避免精度瓶颈掩盖算法逻辑
    q, k, v = q.contiguous().float(), k.contiguous().float(), v.contiguous().float()
    
    B, H, _, D = q.shape
    _, _, L, _ = k.shape
    
    # 【极客调参位】：每个 SM 处理的 KV Token 数量。
    # 对于 1080Ti (28 SMs)，如果你有 16 个 Head，切 2-4 刀就能让显卡满载。
    BLOCK_KV = 256  
    num_chunks = triton.cdiv(L, BLOCK_KV)
    
    mid_o = torch.empty((B, H, num_chunks, D), device=q.device, dtype=torch.float32)
    mid_lse = torch.empty((B, H, num_chunks), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)

    # Stage 1: 并行打散计算
    grid_stage1 = (B * H, num_chunks)
    flash_decoding_stage1[grid_stage1](
        q, k, v, mid_o, mid_lse,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
        B, H, L, D, BLOCK_KV=BLOCK_KV
    )
    
    # Stage 2: 归约合并
    # 找到大于等于 num_chunks 的最小 2 的幂次，Triton 编译器要求
    BLOCK_CHUNKS = triton.next_power_of_2(num_chunks)
    if BLOCK_CHUNKS < 16:
        BLOCK_CHUNKS = 16
        
    grid_stage2 = (B * H, )
    flash_decoding_stage2[grid_stage2](
        mid_o, mid_lse, out,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
        out.stride(0), out.stride(1), out.stride(3),
        B, H, num_chunks, D, BLOCK_CHUNKS=BLOCK_CHUNKS
    )
            
    return out

# --- Baseline & Benchmark ---

def pytorch_native_decoding(q, k, v):
    # PyTorch 原生 Attention 
    # q: [B, H, 1, D], k: [B, H, L, D]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

def benchmark_decoding():
    torch.manual_seed(42)
    print("=" * 50)
    print("🚀 System2 Flash Decoding Profiler (Optimized for Pascal/1080Ti)")
    print("=" * 50)
    
    # 测试极端上下文：Decoding 阶段 Q=1, KV极长
    configs = [
        {'B': 1, 'H': 16, 'L': 8192, 'D': 64},    # 中等 KV Cache
        {'B': 1, 'H': 16, 'L': 65536, 'D': 64},   # 极限长文本 KV Cache (64K)
    ]

    for cfg in configs:
        B, H, L, D = cfg['B'], cfg['H'], cfg['L'], cfg['D']
        print(f"\n[Config] Batch: {B}, Heads: {H}, SeqLen: {L}, HeadDim: {D}")
        
        # 1080Ti 专属 FP32 测试数据
        q = torch.randn((B, H, 1, D), device='cuda', dtype=torch.float32)
        k = torch.randn((B, H, L, D), device='cuda', dtype=torch.float32)
        v = torch.randn((B, H, L, D), device='cuda', dtype=torch.float32)

        # 正确性验证
        out_torch = pytorch_native_decoding(q, k, v)
        out_triton = call_flash_decoding(q, k, v)
        max_diff = (out_torch - out_triton).abs().max().item()
        print(f"  ├─ Correctness Check (Max Diff): {max_diff:.6f} " + ("✅" if max_diff < 1e-3 else "❌"))

        # 测速闭包
        def run_torch(): pytorch_native_decoding(q, k, v)
        def run_triton(): call_flash_decoding(q, k, v)

        # 预热
        for _ in range(10): 
            run_torch()
            run_triton()
        torch.cuda.synchronize()

        # 开始基准测试 (Decoding 速度极快，可循环 1000 次取平均)
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
            times = sorted(times)[100:-100] # 去掉异常值
            avg_time = sum(times) / len(times)
            results[name] = avg_time

        # 分析 Memory Bandwidth
        # 在 Decoding 阶段，每次计算必须读一次 Q(可忽略), 完整的 K 和 V, 写一次 O(可忽略)
        # 所以主要的 HBM I/O = 2 * (B * H * L * D) * 4 Bytes (FP32)
        total_bytes = 2 * B * H * L * D * 4 
        
        for name, t_ms in results.items():
            bw_gbps = (total_bytes / 1e9) / (t_ms / 1000.0)
            print(f"  ├─ [{name}] Time: {t_ms:.3f} ms | Bandwidth: {bw_gbps:.2f} GB/s")

if __name__ == "__main__":
    benchmark_decoding()