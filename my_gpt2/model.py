import numpy as np

def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    Using the approximation used in the original GPT-2 implementation.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Using numerical stability trick by subtracting the max value.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(q, k, v, mask=None):
    """
    Scaled Dot-Product Attention.
    q: query (..., seq_len, head_size)
    k: key (..., seq_len, head_size)
    v: value (..., seq_len, head_size)
    mask: causal mask (seq_len, seq_len)
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e10, scores)
    
    probs = softmax(scores)
    return np.matmul(probs, v)

def mha(x, w_qkv, b_qkv, w_out, b_out, n_head):
    """
    Multi-Head Attention.
    x: input tensor (batch_size, seq_len, embed_dim)
    w_qkv: combined weights for q, k, v (embed_dim, 3 * embed_dim)
    b_qkv: combined biases for q, k, v (3 * embed_dim)
    w_out: output projection weights (embed_dim, embed_dim)
    b_out: output projection bias (embed_dim)
    n_head: number of attention heads
    """
    batch_size, seq_len, embed_dim = x.shape
    qkv = np.matmul(x, w_qkv) + b_qkv
    
    q, k, v = np.split(qkv, 3, axis=-1)
    head_size = embed_dim // n_head
    
    def split_heads(tensor):
        return tensor.reshape(batch_size, seq_len, n_head, head_size).transpose(0, 2, 1, 3)
    
    q, k, v = map(split_heads, [q, k, v])
    mask = np.tril(np.ones((seq_len, seq_len)))
    out = attention(q, k, v, mask=mask)
    
    out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
    return np.matmul(out, w_out) + b_out

def mlp(x, w_fc, b_fc, w_proj, b_proj):
    """
    Feed Forward Network (MLP).
    x: input tensor (batch_size, seq_len, embed_dim)
    w_fc: first linear layer weights (embed_dim, 4 * embed_dim)
    b_fc: first linear layer bias (4 * embed_dim)
    w_proj: second linear layer weights (4 * embed_dim, embed_dim)
    b_proj: second linear layer bias (embed_dim)
    """
    a = gelu(np.matmul(x, w_fc) + b_fc)
    return np.matmul(a, w_proj) + b_proj

class TransformerBlock:
    """
    GPT-2 Transformer Block.
    """
    def __init__(self, params, n_head):
        self.params = params
        self.n_head = n_head
    
    def __call__(self, x):
        # Attention + Residual connection (Pre-LayerNorm)
        x = x + mha(layer_norm(x, **self.params["ln_1"]), **self.params["attn"], n_head=self.n_head)
        # MLP + Residual connection (Pre-LayerNorm)
        x = x + mlp(layer_norm(x, **self.params["ln_2"]), **self.params["mlp"])
        return x

class GPT2:
    """
    GPT-2 Model.
    """
    def __init__(self, params, n_head):
        self.params = params
        self.blocks = [TransformerBlock(p, n_head) for p in params["blocks"]]
    
    def __call__(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # Token + Position Embeddings
        # wte: (vocab_size, embed_dim), wpe: (max_pos, embed_dim)
        x = self.params["wte"][input_ids] + self.params["wpe"][np.arange(input_ids.shape[1])]
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # Final LayerNorm
        x = layer_norm(x, **self.params["ln_f"])
        
        # Language Model Head (Weight Tying)
        # Project back to vocab_size: (batch_size, seq_len, vocab_size)
        return np.matmul(x, self.params["wte"].T)

def layer_norm(x, g, b, eps=1e-5):
    """
    Layer Normalization.
    x: input array
    g: gain (gamma) parameter
    b: bias (beta) parameter
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def main():
    print("GPT-2 basic functions implemented.")

if __name__ == "__main__":
    main()
