import numpy as np
import pytest
from my_gpt2.model import gelu, softmax, layer_norm

def test_gelu():
    # GELU(x) should be approximately 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Test a few key values
    assert np.isclose(gelu(np.array([0.0])), 0.0, atol=1e-5)
    assert gelu(np.array([1.0])) > 0.8  # GELU(1) is approx 0.8413
    assert gelu(np.array([-1.0])) < 0.0 # GELU(-1) is approx -0.1587

def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    probs = softmax(x)
    # Sum should be 1.0
    assert np.isclose(np.sum(probs), 1.0)
    # Order should be preserved
    assert probs[0] < probs[1] < probs[2]
    # Stability: check very large numbers don't overflow
    x_large = np.array([1000.0, 1001.0, 1002.0])
    probs_large = softmax(x_large)
    assert np.isclose(np.sum(probs_large), 1.0)

def test_layer_norm():
    x = np.random.randn(10, 768) # 768 is GPT-2 small embedding size
    # In GPT-2, g (gamma) and b (beta) are parameters
    g = np.ones(768)
    b = np.zeros(768)
    
    out = layer_norm(x, g, b)
    
    # After layer norm, along the last axis, mean should be ~0 and std should be ~1
    assert np.allclose(np.mean(out, axis=-1), 0.0, atol=1e-5)
    assert np.allclose(np.std(out, axis=-1), 1.0, atol=1e-5)
