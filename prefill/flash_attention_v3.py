import torch
import triton
import triton.language as tl

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
