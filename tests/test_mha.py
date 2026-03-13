import numpy as np
import pytest
from my_gpt2.model import mha

def test_mha_shape():
    batch_size, seq_len, embed_dim = 1, 10, 768 # GPT-2 small
    n_head = 12
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Combined weights for Q, K, V (3 * embed_dim)
    w_qkv = np.random.randn(embed_dim, 3 * embed_dim)
    b_qkv = np.zeros(3 * embed_dim)
    # Output projection
    w_out = np.random.randn(embed_dim, embed_dim)
    b_out = np.zeros(embed_dim)
    
    out = mha(x, w_qkv, b_qkv, w_out, b_out, n_head)
    
    assert out.shape == (batch_size, seq_len, embed_dim)

def test_mha_causal():
    # Similar to attention test, but through the MHA wrapper
    batch_size, seq_len, embed_dim = 1, 4, 16
    n_head = 2
    x1 = np.random.randn(batch_size, seq_len, embed_dim)
    
    w_qkv = np.random.randn(embed_dim, 3 * embed_dim)
    b_qkv = np.zeros(3 * embed_dim)
    w_out = np.random.randn(embed_dim, embed_dim)
    b_out = np.zeros(embed_dim)
    
    out1 = mha(x1, w_qkv, b_qkv, w_out, b_out, n_head)
    
    # Modify the last token of input
    x2 = x1.copy()
    x2[:, -1, :] += 10.0
    
    out2 = mha(x2, w_qkv, b_qkv, w_out, b_out, n_head)
    
    # Causal property: first 3 tokens should not change
    assert np.allclose(out1[:, :3, :], out2[:, :3, :])
