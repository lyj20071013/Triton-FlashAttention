import torch
import triton
import triton.language as tl
from timeit import Timer
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from prefill.flash_attention_v1 import call_flash_attention_v1
from prefill.flash_attention_v2 import call_flash_attention_v2
from prefill.flash_attention_v3 import call_flash_attention_v3

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
