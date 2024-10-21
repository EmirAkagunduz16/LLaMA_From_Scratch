import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelArgs:
  dim: int = 4096
  n_layers: int = 32
  n_heads: int = 32 # Number of heads for the queries
  n_kv_heads: Optional[int] = None # Number of heads for the K and V
  vocab_size: int = -1 # This will be set when we load the tokenizer
  multiple_of: int = 256
  ffn_dim_multiplier: Optional[float] = None
  norm_eps: float = 1e-5
  
  # needed for KV cache
  max_batch_size: int = 32
  max_seq_len: int = 2048
  device: str = None
  

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> List:
  assert head_dim % 2 == 0, "Dimension must be divisible by 2"
  # Build the theta parameters
  # According the formula theta_i = 10000 ^ (-2*(i-1)/dim) for i = [1,2,3 ... ,d/2]
  
  # Shape: (Head_dim / 2)
  theta_numerator = torch.arange(0, head_dim, 2).float()
  # Shape: (Head_dim / 2)
  theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
  
  # Construct the positions (the "m" parameter)
  # Shape: (Seq_len)
  m = torch.arange(seq_len, device=device)
  # Multiply each theta by each position using the outer product
  # Shape: (Seq_len) ⊗ (Head_dim / 2) -> (Seq_len, Head_dim / 2)
  freqs = torch.outer(m, theta).float()
  # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
  # (Seq_len, Head_dim/2) -> (Seq_len, Head_dim/2)
  freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_complex

 
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
  # (B, Seq_len, H, Head_dim) -> (B, Seq_len, H, Head_dim/2)
  x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
  # (Seq_len, Head_dim/2) -> (1, Seq_len, 1, Head_dim/2)
  freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
  # (B, Seq_len, H, Head_dim/2) * (1, Seq_len, 1, Head_dim/2) = (B, Seq_len, H, Head_dim/2)
  x_rotated = x_complex * freqs_complex
  # (B, Seq_len, H, Head_dim/2) -> (B, Seq_len, H, Head_dim/2, 2)
  x_out = torch.view_as_real(x_rotated)
  # (B, Seq_len, H, Head_dim/2, 2) -> (B, Seq_len, H, Head_dim/2)
  x_out = x_out.reshape(*x.shape)
  return x_out.type_as(x).to(device) 


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
  batch_size, seq_len, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  else:
    return(
      # (B, seq_len, N_KV_heads, 1, Head_dim)
      x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )  
  
  
class RMSNorm(nn.Module):
  
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps     
    # The gamma parameter
    self.weight = nn.Parameter(torch.ones(dim))
  
  def _norm(self, x: torch.Tensor):
    # (B, Seq_len, Dim) * (B, Seq_len, 1) = (B, Seq_len, Dim)
    # rsqrt= 1 / sqrt(x)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: torch.Tensor):
    # (Dim) * (B, Seq_len, Dim) = (B, Seq_len, Dim)
    return self.weight * self._norm(x.float()).type_as(x)  
  

class EncoderBlock(nn.Module):
  
  def __init__(self, args: ModelArgs):
    super().__init__()
    
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    
    self.attention = SelfAttention(args)
    self.feed_forward = FeedForward(args)
    
    # Normalization BEFORE Self Attention
    self.attention_norm = RMSNorm(args.dim, args.norm_eps)
    # Normalization BEFORE Feed Forward
    self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
  
  def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
    # (B, Seq_len, Dim) + (B, Seq_len, Dim) -> (B, Seq_len, Dim) 
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out
  
    
class SelfAttention(nn.Module):
  
  def __init__(self, args: ModelArgs):
    super().__init__()
    # Indicates the number of heads for the Key and Values 
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    # Indicates the number of heads for the Queries 
    self.n_heads_q = args.n_heads
    # Indicates how many times the heads of Keys and Values should be repeated to match the head of Queries
    self.n_rep = self.n_heads_q // self.n_kv_heads
    # Indicates the dimension of each head
    self.head_dim = args.dim // args.n_heads
    # 
    self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
    
    self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
  
  def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
    batch_size, seq_len, _ = x.shape # (B, 1, Dim)
    
    # Apply the Wq, Wk and Wv matrices to queries, keys and values 
    # (B, 1, Dim) -> (B, 1, H_Q, Head_dim)
    xq = self.wq(x)
    # (B, 1, Dim) -> (B, 1, H_KV, Head_dim)
    xk = self.wk(x)
    xv = self.wv(x)
    
    # (B, 1, H_Q, Head_dim) -> (B, 1, H_Q, Head_dim)
    xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
    # (B, 1, H_KV, Head_dim) -> (B, 1, H_KV, Head_dim)
    xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
    xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
    
    # Does not change the shape of tensors
    xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
    xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
    
    # replace the entry in the cache for this token
    self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
    self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
    
    # Retrieve all the cached keys and values so far
    # (B, Seq_len_KV, H_KV, Head_dim)
    keys = self.cache_k[:batch_size, :start_pos+seq_len]
    values = self.cache_v[:batch_size, :start_pos+seq_len]
    
    # Repeat the heads of the K and V to reach the number of heads of the queries.
    keys = repeat_kv(keys, self.n_rep)
    values = repeat_kv(values, self.n_rep)
    
    # (B, 1, H_Q, Head_dim) -> (B, H_Q, 1, Head_dim)
    xq = xq.transpose(1, 2)
    # (B, H_Q, Seq_len_KV, Head_dim)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    
    # (B, 1, H_Q, Head_dim) @ (B, H_Q, Head_dim, Seq_len_KV) = (B, H_Q, 1, Seq_len_KV)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    
    # (B, H_Q, 1, Seq_len_KV) @ (B, H_Q, Seq_len_KV, Head_dim) = (B, H_Q, 1, Head_dim)
    out = torch.matmul(scores, values)
    
    # (B, H_Q, 1, Head_dim) -> (B, 1, H_Q, Head_dim) -> (B, 1, H_Q * Head_dim) -> (B, 1, Dim)
    out = (out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
    
    # (B, 1, Dim) -> (B, 1, Dim)
    return self.wo(out)
     
  
class FeedForward(nn.Module):
  
  def __init__(self, args: ModelArgs):
    super().__init__()
    
    hidden_dim = 4 * args.dim
    hidden_dim = int(2 * hidden_dim / 3)
    if args.ffn_dim_multiplier is not None: hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
    # Round the hidden dim to the nearest of mutliple_of parameter
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
    
    self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
    self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
  
  def forward(self, x: torch.Tensor):
    return self.w2(F.silu(self.w1(x)) * self.w3(x)) 
  
  
class Transformer(nn.Module):
  
  def __init__(self, args: ModelArgs) -> None:
    super().__init__()
    
    assert args.vocab_size != -1, "Vocab size must be set"
    
    self.args = args
    self.vocab_size = args.vocab_size
    self.n_layers = args.n_layers
    self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
    
    self.layers = nn.ModuleList()
    for _ in range(self.n_layers):
      self.layers.append(EncoderBlock(args))
    
    self.norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
    
    self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=args.device)
    
  def forward(self, tokens: torch.Tensor, start_pos: int):
    # (B, Seq_Len)
    batch, seq_len = tokens.shape
    assert seq_len == 1, "Only one token at a time can be processed"
    
    # (B, Seq_len) -> (B, Seq_Len, Dim)
    h = self.tok_embeddings(tokens)
    
    # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
    freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
    
    # Consecutevily apply all the encoder layers
    for layer in self.layers:
      h = layer(h, start_pos, freqs_complex)
    h = self.norm(h)
    output = self.output(h).float()
    return output