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
