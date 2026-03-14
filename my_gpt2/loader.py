import numpy as np
from safetensors.numpy import load_file

def load_gpt2_weights(model_id="openai-community/gpt2"):
    """
    GPT-2のsafetensorsを読み込み、paramsディクショナリにマッピングする。
    事前に `make download` を実行して重みファイルを取得してください。
    """
    file_path = f"weights/{model_id}/model.safetensors"

    # 1. 重みをnumpy配列として読み込む
    print(f"{file_path} からメモリに重みを読み込み中...")
    weights = load_file(file_path)

    # 2. paramsの構造にマッピング
    prefix = "transformer." if "transformer.wte.weight" in weights else ""

    params = {
        "wte": weights[f"{prefix}wte.weight"],
        "wpe": weights[f"{prefix}wpe.weight"],
        "ln_f": {"g": weights[f"{prefix}ln_f.weight"], "b": weights[f"{prefix}ln_f.bias"]},
        "blocks": []
    }

    n_layer = sum(1 for key in weights.keys() if key.endswith(".ln_1.weight"))
    print(f"トランスフォーマーブロック数: {n_layer}")

    for i in range(n_layer):
        block_params = {
            "ln_1": {
                "g": weights[f"{prefix}h.{i}.ln_1.weight"],
                "b": weights[f"{prefix}h.{i}.ln_1.bias"]
            },
            "attn": {
                # GPT-2のConv1D重みは既に(embed_dim, out_dim)形式なので
                # x @ w_qkv をそのまま使える。転置は不要。
                "w_qkv": weights[f"{prefix}h.{i}.attn.c_attn.weight"],
                "b_qkv": weights[f"{prefix}h.{i}.attn.c_attn.bias"],
                "w_out": weights[f"{prefix}h.{i}.attn.c_proj.weight"],
                "b_out": weights[f"{prefix}h.{i}.attn.c_proj.bias"],
            },
            "ln_2": {
                "g": weights[f"{prefix}h.{i}.ln_2.weight"],
                "b": weights[f"{prefix}h.{i}.ln_2.bias"]
            },
            "mlp": {
                "w_fc": weights[f"{prefix}h.{i}.mlp.c_fc.weight"],
                "b_fc": weights[f"{prefix}h.{i}.mlp.c_fc.bias"],
                "w_proj": weights[f"{prefix}h.{i}.mlp.c_proj.weight"],
                "b_proj": weights[f"{prefix}h.{i}.mlp.c_proj.bias"],
            }
        }
        params["blocks"].append(block_params)

    return params
