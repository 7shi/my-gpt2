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
    # q @ k.T / sqrt(d_k)
    # k is (..., seq_len, head_size), transpose it to (..., head_size, seq_len)
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    if mask is not None:
        # mask is (seq_len, seq_len), where 1 means keep and 0 means mask out.
        # fill 0s with a very large negative number
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
    
    # Linear transformation for Q, K, V
    qkv = np.matmul(x, w_qkv) + b_qkv # (batch_size, seq_len, 3 * embed_dim)
    
    # Split Q, K, V and reshape to (batch_size, n_head, seq_len, head_size)
    q, k, v = np.split(qkv, 3, axis=-1)
    head_size = embed_dim // n_head
    
    def split_heads(tensor):
        # (batch_size, seq_len, embed_dim) -> (batch_size, n_head, seq_len, head_size)
        return tensor.reshape(batch_size, seq_len, n_head, head_size).transpose(0, 2, 1, 3)
    
    q, k, v = map(split_heads, [q, k, v])
    
    # Causal mask
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    # Core attention
    out = attention(q, k, v, mask=mask) # (batch_size, n_head, seq_len, head_size)
    
    # Merge heads
    out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
    
    # Output projection
    return np.matmul(out, w_out) + b_out

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
