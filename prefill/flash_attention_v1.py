import torch
import triton
import triton.language as tl

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
