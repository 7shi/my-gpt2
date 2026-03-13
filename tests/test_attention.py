import numpy as np
import pytest
from my_gpt2.model import attention

def test_attention_shape():
    batch_size, n_head, seq_len, head_size = 1, 12, 10, 64
    q = np.random.randn(batch_size, n_head, seq_len, head_size)
    k = np.random.randn(batch_size, n_head, seq_len, head_size)
    v = np.random.randn(batch_size, n_head, seq_len, head_size)
    
    out = attention(q, k, v, mask=None)
    assert out.shape == (batch_size, n_head, seq_len, head_size)

def test_causal_mask():
    batch_size, n_head, seq_len, head_size = 1, 1, 4, 8
    q = np.random.randn(batch_size, n_head, seq_len, head_size)
    k = np.random.randn(batch_size, n_head, seq_len, head_size)
    v = np.random.randn(batch_size, n_head, seq_len, head_size)
    
    # causal mask (lower triangular)
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    out1 = attention(q, k, v, mask=mask)
    
    # Modify the last token of k and v
    k_mod = k.copy()
    v_mod = v.copy()
    k_mod[:, :, -1, :] += 10.0
    v_mod[:, :, -1, :] += 10.0
    
    out2 = attention(q, k_mod, v_mod, mask=mask)
    
    # The output for the first 3 tokens should be identical
    # because they shouldn't see the 4th token due to the mask.
    assert np.allclose(out1[:, :, :3, :], out2[:, :, :3, :])
    # The last token output should be different
    assert not np.allclose(out1[:, :, -1, :], out2[:, :, -1, :])
