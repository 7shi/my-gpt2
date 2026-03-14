import numpy as np
import pytest
from my_gpt2.model import mha

def test_mha_shape():
    batch_size, seq_len, embed_dim = 1, 10, 768 # GPT-2 small
    n_head = 12
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Q, K, V の結合重み (3 * embed_dim)
    w_qkv = np.random.randn(embed_dim, 3 * embed_dim)
    b_qkv = np.zeros(3 * embed_dim)
    # 出力射影
    w_out = np.random.randn(embed_dim, embed_dim)
    b_out = np.zeros(embed_dim)

    out = mha(x, w_qkv, b_qkv, w_out, b_out, n_head)

    assert out.shape == (batch_size, seq_len, embed_dim)

def test_mha_causal():
    # アテンションテストと同様だが、MHAラッパーを通した確認
    batch_size, seq_len, embed_dim = 1, 4, 16
    n_head = 2
    x1 = np.random.randn(batch_size, seq_len, embed_dim)

    w_qkv = np.random.randn(embed_dim, 3 * embed_dim)
    b_qkv = np.zeros(3 * embed_dim)
    w_out = np.random.randn(embed_dim, embed_dim)
    b_out = np.zeros(embed_dim)

    out1 = mha(x1, w_qkv, b_qkv, w_out, b_out, n_head)

    # 入力の最後のトークンを変更
    x2 = x1.copy()
    x2[:, -1, :] += 10.0

    out2 = mha(x2, w_qkv, b_qkv, w_out, b_out, n_head)

    # 因果性: 最初の3トークンは変化しないはず
    assert np.allclose(out1[:, :3, :], out2[:, :3, :])
