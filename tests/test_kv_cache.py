import numpy as np
from my_gpt2.model import GPT2, LayerNormParams, AttentionParams, MLPParams, BlockParams, GPT2Params


def make_model(vocab_size=100, embed_dim=16, n_head=2, n_layer=2, max_pos=128):
    np.random.seed(42)
    params = GPT2Params(
        wte=np.random.randn(vocab_size, embed_dim),
        wpe=np.random.randn(max_pos, embed_dim),
        blocks=[
            BlockParams(
                ln_1=LayerNormParams(g=np.ones(embed_dim), b=np.zeros(embed_dim)),
                attn=AttentionParams(
                    w_qkv=np.random.randn(embed_dim, 3 * embed_dim),
                    b_qkv=np.zeros(3 * embed_dim),
                    w_out=np.random.randn(embed_dim, embed_dim),
                    b_out=np.zeros(embed_dim),
                ),
                ln_2=LayerNormParams(g=np.ones(embed_dim), b=np.zeros(embed_dim)),
                mlp=MLPParams(
                    w_fc=np.random.randn(embed_dim, 4 * embed_dim),
                    b_fc=np.zeros(4 * embed_dim),
                    w_proj=np.random.randn(4 * embed_dim, embed_dim),
                    b_proj=np.zeros(embed_dim),
                ),
            )
            for _ in range(n_layer)
        ],
        ln_f=LayerNormParams(g=np.ones(embed_dim), b=np.zeros(embed_dim)),
    )
    return GPT2(params, n_head)


def test_kv_cache_equivalence():
    """キャッシュあり/なしで最終トークンの logits が一致する。"""
    model = make_model()
    input_ids = np.array([10, 20, 30, 40, 50])

    # キャッシュなし
    logits_no_cache = model(input_ids)

    # Prefill + 1 step incremental
    prefill_ids = np.array([10, 20, 30, 40])
    _, kv_cache = model(prefill_ids, kv_cache=None)
    last_id = np.array([50])
    logits_cached, _ = model(last_id, kv_cache=kv_cache)

    np.testing.assert_allclose(
        logits_cached[0, :], logits_no_cache[-1, :], atol=1e-5
    )


def test_kv_cache_multistep():
    """複数ステップの greedy 生成でキャッシュあり/なしが一致する。"""
    model = make_model()
    prompt = [10, 20, 30]
    n_steps = 5

    # キャッシュなし
    ids_no_cache = prompt.copy()
    for _ in range(n_steps):
        logits = model(np.array(ids_no_cache))
        ids_no_cache.append(int(np.argmax(logits[-1, :])))

    # キャッシュあり
    ids_cached = prompt.copy()
    logits, kv_cache = model(np.array(prompt), kv_cache=None)
    ids_cached.append(int(np.argmax(logits[-1, :])))
    for _ in range(n_steps - 1):
        logits, kv_cache = model(np.array([ids_cached[-1]]), kv_cache=kv_cache)
        ids_cached.append(int(np.argmax(logits[-1, :])))

    assert ids_no_cache == ids_cached
