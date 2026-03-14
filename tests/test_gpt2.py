import numpy as np
import pytest
from my_gpt2.model import gelu, softmax, layer_norm, LayerNormParams

def test_gelu():
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # いくつかの代表的な値でテスト
    assert np.isclose(gelu(np.array([0.0])), 0.0, atol=1e-5)
    assert gelu(np.array([1.0])) > 0.8  # GELU(1) ≈ 0.8413
    assert gelu(np.array([-1.0])) < 0.0 # GELU(-1) ≈ -0.1587

def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    probs = softmax(x)
    # 合計が1.0になるはず
    assert np.isclose(np.sum(probs), 1.0)
    # 順序が保たれるはず
    assert probs[0] < probs[1] < probs[2]
    # 数値安定性: 非常に大きな値でオーバーフローしないことを確認
    x_large = np.array([1000.0, 1001.0, 1002.0])
    probs_large = softmax(x_large)
    assert np.isclose(np.sum(probs_large), 1.0)

def test_layer_norm():
    x = np.random.randn(10, 768) # 768はGPT-2 smallの埋め込みサイズ
    # GPT-2では g（ガンマ）と b（ベータ）がパラメータ
    params = LayerNormParams(g=np.ones(768), b=np.zeros(768))

    out = layer_norm(x, params)

    # レイヤー正規化後、最終軸方向の平均は約0、標準偏差は約1になるはず
    assert np.allclose(np.mean(out, axis=-1), 0.0, atol=1e-5)
    assert np.allclose(np.std(out, axis=-1), 1.0, atol=1e-5)
