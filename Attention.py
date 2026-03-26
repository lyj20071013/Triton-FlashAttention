import torch
import triton
import triton.language as tl
from timeit import Timer
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

@triton.jit
def flash_attention_v1(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, head_dim: tl.constexpr, 
    stride_qm, stride_qh, 
    stride_km, stride_kh,
    stride_vm, stride_vh,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    # 2D Grid parallel: partition over sequence (0) and heads (1)
    pid_m = tl.program_id(0)  
    pid_head = tl.program_id(1) 

    start_m = pid_m * BLOCK_M

    # Generate local offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)

    # Offset base pointers to the current head
    q_ptr = q_ptr + pid_head * stride_qh
    k_ptr = k_ptr + pid_head * stride_kh
    v_ptr = v_ptr + pid_head * stride_vh
    o_ptr = o_ptr + pid_head * stride_oh
    
    # Initialize accumulators
    m_prev = tl.full((BLOCK_M, ), float('-inf'), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    # Load Q block [BLOCK_M, head_dim]
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0
    ).to(tl.float32)
    
    # Iterate over K/V blocks
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_m = offs_m < seq_len  
        mask_n = offs_n < seq_len  

        k = tl.load(
            k_ptr + offs_n[:, None] * stride_km + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0
        ).to(tl.float32)
        
        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vm + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0
        ).to(tl.float32)
        
        # Compute QK^T
        s = tl.dot(q, k.T)
        s *= 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None]) >= (start_n + offs_n[None, :])
            s = tl.where(causal_mask, s, float('-inf'))

        # Online Softmax 
        m_curr = tl.maximum(tl.max(s, axis=1), m_prev)
        alpha = tl.exp(m_prev - m_curr)
        beta = tl.exp(s - m_curr[:, None])
        l_curr = alpha * l_prev + tl.sum(beta, axis=1)
        
        p = beta 
        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_prev = m_curr
        l_prev = l_curr

    # Epilogue normalization
    acc = acc / l_prev[:, None]

    # Store final result
    tl.store(
        o_ptr + offs_m[:, None] * stride_om + offs_d[None, :],
        acc,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    )

def call_flash_attention_v1(q, k, v, is_causal=False):
    assert q.dim() == 3, "Input should be [seq_len, num_heads, head_dim]"
    seq_len, num_heads, head_dim = q.shape
    o = torch.empty_like(q)

    BLOCK_M, BLOCK_N = 32, 32
    
    # Launch 2D grid
    grid = (triton.cdiv(seq_len, BLOCK_M), num_heads)

    flash_attention_v1[grid](
        q, k, v, o,
        seq_len, head_dim,
        q.stride(0), q.stride(1),  
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal
    )
    return o


@triton.jit
def flash_attention_v2(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, head_dim: tl.constexpr,
    q_stride_m, q_stride_h, 
    k_stride_m, k_stride_h, 
    v_stride_m, v_stride_h, 
    o_stride_m, o_stride_h, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    NUM_HEADS: tl.constexpr, 
    IS_CAUSAL: tl.constexpr 
):
    pid_head = tl.program_id(0) 
    pid_m = tl.program_id(1) 

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulators
    m_i = tl.full((BLOCK_M, ), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    # Load Q block with vectorized loading
    q_offset = pid_head * q_stride_h + offs_m[:, None] * q_stride_m
    q = tl.load(q_ptr + q_offset + tl.arange(0, head_dim)[None, :] * q_stride_h,
                mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim),
                other=0.0).to(tl.float32)
    
    # Main loop over K/V blocks
    for start_n in range(0, (seq_len + BLOCK_M - 1) // BLOCK_N * BLOCK_N, BLOCK_N):
        valid_n = start_n + offs_n < seq_len
        
        k_offset = pid_head * k_stride_h + (start_n + offs_n)[:, None] * k_stride_m
        k = tl.load(k_ptr + k_offset + tl.arange(0, head_dim)[None, :] * 1,
                    mask=valid_n[:, None] & (tl.arange(0, head_dim)[None, :] < head_dim), 
                    other=0.0).to(tl.float32)
        
        v_offset = pid_head * v_stride_h + (start_n + offs_n)[:, None] * v_stride_m
        v = tl.load(v_ptr + v_offset + tl.arange(0, head_dim)[None, :] * 1,
                    mask=valid_n[:, None] & (tl.arange(0, head_dim)[None, :] < head_dim),
                    other=0.0).to(tl.float32)
        
        # QK^T utilizing Tensor Cores
        s = tl.dot(q, k.T.to(q.dtype))
        s = s * (1.0 / tl.sqrt(tl.cast(head_dim, tl.float32)))

        if IS_CAUSAL:
            causal_mask = (offs_m[:, None]) >= (start_n + offs_n[None, :])
            s = tl.where(causal_mask, s, float('-inf'))

        # Online Softmax update
        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        alpha = tl.exp(m_i - m_curr) 
        beta = tl.exp(s - m_curr[:, None]) 
        l_curr = alpha * l_i + tl.sum(beta, axis=1)
        
        p = beta 
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_curr
        l_i = l_curr

    # Epilogue normalization
    o = acc / l_i[:, None]
    
    # Store to global memory
    o_offset = pid_head * o_stride_h + offs_m[:, None] * o_stride_m
    tl.store(o_ptr + o_offset + tl.arange(0, head_dim)[None, :] * 1,
             o.to(o_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim))

def call_flash_attention_v2(q, k, v, is_causal=False):
    assert q.dim() == 3, "Input should be [seq_len, num_heads, head_dim]"
    seq_len, num_heads, head_dim = q.shape
    o = torch.empty_like(q)
    
    config = {
        'BLOCK_M': 128,
        'BLOCK_N': 64,
        'num_warps': 8,
        'num_stages': 3,
    }
    
    # Grid dimensions: [num_heads, num_Q_blocks]
    grid = (num_heads, triton.cdiv(seq_len, config['BLOCK_M']))
    
    flash_attention_v2[grid](
        q, k, v, o,
        seq_len, head_dim,
        q.stride(1), q.stride(0),
        k.stride(1), k.stride(0),
        v.stride(1), v.stride(0),
        o.stride(1), o.stride(0),
        NUM_HEADS=num_heads,
        IS_CAUSAL=is_causal,
        **config
    )
    return o


@triton.jit
def flash_attention_v3(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, head_dim: tl.constexpr,
    stride_qm, stride_qh,
    stride_km, stride_kh,
    stride_vm, stride_vh,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    USE_FP8: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    pid = tl.program_id(0)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulators (adjust precision based on quantization mode)
    acc_dtype = tl.float8e5 if USE_FP8 else tl.float32
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=acc_dtype)

    # Load Q block
    if USE_FP8:
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + tl.arange(0, head_dim)[None, :] * stride_qh,
            mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim),
            other=0.0
        ).to(tl.float8e5)
    else:
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + tl.arange(0, head_dim)[None, :] * stride_qh,
            mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim),
            other=0.0
        ).to(tl.float32)

    # Initialize K/V block pointers
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(seq_len, head_dim),
        strides=(stride_km, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, head_dim),
        order=(0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(seq_len, head_dim),
        strides=(stride_vm, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, head_dim),
        order=(0, 1)
    )

    # Main loop
    for start_n in range(0, seq_len, BLOCK_N):
        curr_k = tl.load(k_block_ptr, boundary_check=(0,)).to(tl.float8e5 if USE_FP8 else tl.float32)
        curr_v = tl.load(v_block_ptr, boundary_check=(0,)).to(tl.float8e5 if USE_FP8 else tl.float32)

        if USE_FP8:
            s = tl.dot(q, tl.trans(curr_k), allow_tf32=True).to(tl.float32)
        else:
            s = tl.dot(q.to(tl.float32), tl.trans(curr_k.to(tl.float32)))
        
        s = s * (1.0 / tl.sqrt(tl.cast(head_dim, tl.float32)))

        if IS_CAUSAL:
            causal_mask = (offs_m[:, None]) >= (start_n + offs_n[None, :])
            s = tl.where(causal_mask, s, float('-inf'))
            
        # Online Softmax
        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        alpha = tl.exp(m_i - m_curr)
        beta = tl.exp(s - m_curr[:, None])
        l_curr = alpha * l_i + tl.sum(beta, axis=1)
        
        p = beta 

        # Accumulate output (upcast to FP32)
        if USE_FP8:
            p_fp32 = p.to(tl.float32)
            curr_v_fp32 = curr_v.to(tl.float32) 
            acc = acc * alpha[:, None] + tl.dot(p_fp32, curr_v_fp32)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, curr_v.to(tl.float32))

        m_i = m_curr
        l_i = l_curr

        # Prefetch next block
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    # Epilogue normalization
    acc = acc / l_i[:, None]

    # Store results (maintain FP8 format if specified)
    tl.store(
        o_ptr + offs_m[:, None] * stride_qm + tl.arange(0, head_dim)[None, :],
        acc.to(tl.float8e5 if USE_FP8 else tl.float32),
        mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim)
    )

def call_flash_attention_v3(q, k, v, use_fp8=False, is_causal=False):
    assert q.dim() == 3, "Input should be [seq_len, num_heads, head_dim]"
    
    # Enforce FP8 formats for inputs/outputs (PyTorch 2.1+)
    if use_fp8:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.to(torch.float8_e5m2)
        o = torch.empty_like(q, dtype=torch.float8_e5m2)
    else:
        o = torch.empty_like(q)
        
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "USE_FP8": use_fp8,
        "num_warps": 8,
        "num_stages": 3
    }

    grid = (triton.cdiv(q.size(0), config['BLOCK_M']),)
    flash_attention_v3[grid](
        q, k, v, o,
        q.size(0), q.size(-1),
        q.stride(1), q.stride(0),
        k.stride(1), k.stride(0),
        v.stride(1), v.stride(0),
        IS_CAUSAL=is_causal,
        **config
    )
    return o

# --- PyTorch Native Implementations ---
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1) 
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k)) 
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) 

        attn_weights = F.softmax(scores, dim=-1) 
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, value) 
        return output, attn_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model) 
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.W_q(q)) 
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) 
        
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        batch_size, _, seq_len, _ = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        
        return output, attn_weights
    
# --- Global Setup ---
block_size = 1024
n_embd = 64
dropout = 0.1

class Head(nn.Module):
    def __init__(self, head_size, causal=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.causal = causal
        if self.causal:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x) 
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        if self.causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei) 
        v = self.value(x) 
        out = wei @ v 
        return out

def pytorch_attention(q, k, v, causal, head_size):
    x = q.unsqueeze(0) 
    mha = Head(head_size=head_size, causal=causal)
    mha = mha.to(q.device)
    output = mha(x)
    return output.squeeze(0) 

def plot(times, memory_usage):
    methods = ['Pytorch Native', 'FlashAttention-v1', 'FlashAttention-v2', 'FlashAttention-v3']

    plt.figure(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width/2, times, width, label='Time (ms)', color='#1f77b4')  
    plt.bar(x + width/2, memory_usage, width, label='Memory Usage (MB)', color='#ff7f0e') 

    plt.xlabel('Method', fontsize=12, fontfamily='sans-serif')  
    plt.ylabel('Value', fontsize=12, fontfamily='sans-serif')
    plt.title('Comparison of Kernel Times and Memory Usage', fontsize=14, fontfamily='sans-serif')

    plt.xticks(x, methods, fontsize=10, fontfamily='sans-serif')  
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  

    for i, (acc, mem) in enumerate(zip(times, memory_usage)):
        plt.text(i - width/2, acc + 0.5, f'{acc:.2f} ms', ha='center', va='bottom', fontsize=10, fontfamily='sans-serif')
        plt.text(i + width/2, mem + 0.5, f'{mem:.2f} MB', ha='center', va='bottom', fontsize=10, fontfamily='sans-serif')

    plt.grid(True, linestyle='--', alpha=0.7)  
    plt.tight_layout()  

    plt.savefig("acc.png", dpi=300)  
    plt.show()

def benchmark_fn(fn, args, num_repeats=100):
    # Warmup
    for _ in range(20): 
        fn(*args)
    torch.cuda.synchronize() 
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    
    for _ in range(num_repeats):
        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize() 
        
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    
    # Dynamic Trimming: Remove top and bottom 20%
    times = sorted(times)
    trim_count = int(num_repeats * 0.2)
    if trim_count > 0:
        trimmed_times = times[trim_count:-trim_count]
    else:
        trimmed_times = times
    return sum(trimmed_times) / len(trimmed_times)

def profile_hardware_metrics(seq_len, num_heads, head_dim, time_ms, is_causal=True, dtype_size=2):
    flops_per_head = 4 * (seq_len ** 2) * head_dim
    if is_causal:
        flops_per_head /= 2 
    
    total_flops = flops_per_head * num_heads
    tflops = (total_flops / 1e12) / (time_ms / 1000.0)

    total_bytes = 4 * seq_len * num_heads * head_dim * dtype_size
    bandwidth_gbps = (total_bytes / 1e9) / (time_ms / 1000.0)
    
    intensity = total_flops / total_bytes

    print(f"  ├─ Hardware Performance:")
    print(f"  │  ├─ TFLOPs/s: {tflops:.2f} TFLOPs")
    print(f"  │  ├─ Bandwidth: {bandwidth_gbps:.2f} GB/s")
    print(f"  │  └─ Arithmetic Intensity: {intensity:.2f} FLOPs/Byte")

def benchmark_attention():
    torch.manual_seed(42)
    configs = [
        {'seq_len': 1024, 'd_model': 128, 'num_heads': 8},
        {'seq_len': 8192, 'd_model': 256, 'num_heads': 16},
    ]

    for cfg in configs:
        global n_embd, block_size
        n_embd, block_size = cfg['d_model'], cfg['seq_len']
        print(f"\n=========================================")
        print(f"Benchmarking config: {cfg}")
        print(f"=========================================")
        
        q_single = torch.randn(cfg['seq_len'], cfg['d_model'], device='cuda', dtype=torch.float32)
        k_single, v_single = torch.randn_like(q_single), torch.randn_like(q_single)
        head_size = cfg['d_model'] // cfg['num_heads']
        
        q_multi = torch.randn(cfg['seq_len'], cfg['num_heads'], head_size, device='cuda', dtype=torch.float16)
        k_multi = torch.randn_like(q_multi)
        v_multi = torch.randn_like(q_multi)

        test_cases = [
            ('PyTorch Native', pytorch_attention, (q_single, k_single, v_single, False, head_size)), 
            ('FlashAttention-v1', call_flash_attention_v1, (q_multi, k_multi, v_multi, True)),
            ('FlashAttention-v2', call_flash_attention_v2, (q_multi, k_multi, v_multi, True)),
            ('FlashAttention-v3', call_flash_attention_v3, (q_multi, k_multi, v_multi, False, True)),
        ]
        
        results = {}
        mem_usage = {}
        for name, fn, args in test_cases:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            time_ms = benchmark_fn(fn, args, num_repeats=100) 
            results[name] = time_ms
            
            fn(*args)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            mem_usage[name] = peak_mem
        
        times = []
        memory_usage = []
        for name in results.keys():
            time = results[name]
            mem = mem_usage[name]
            times.append(time)
            memory_usage.append(mem)
            
            print(f"\n[{name}]")
            print(f"  ├─ Time: {time:.3f} ms")
            print(f"  ├─ Peak Memory: {mem:.2f} MB")
            
            if 'PyTorch' not in name:
                profile_hardware_metrics(cfg['seq_len'], cfg['num_heads'], head_size, time, is_causal=True, dtype_size=2)
        
        plot(times, memory_usage)

if __name__ == "__main__":
    benchmark_attention()
