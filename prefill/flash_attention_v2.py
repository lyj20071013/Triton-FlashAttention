import torch
import triton
import triton.language as tl

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