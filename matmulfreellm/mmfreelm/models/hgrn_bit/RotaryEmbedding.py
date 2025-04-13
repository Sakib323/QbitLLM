class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) optimized for ternary weights"""
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self._build_cache()
    
    def _build_cache(self):
        seq = torch.arange(self.max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.bfloat16), persistent=False)
    
    def _apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        """Ternary-optimized rotary transformation"""
        seq_len = x.size(1)
        cos = self.cos_cached[:seq_len].view(1, seq_len, 1, self.dim)
        sin = self.sin_cached[:seq_len].view(1, seq_len, 1, self.dim)
        
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_rotary(x)