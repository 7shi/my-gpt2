import numpy as np
from safetensors.numpy import load_file
from my_gpt2.model import LayerNormParams, AttentionParams, MLPParams, BlockParams, GPT2Params

def load_gpt2_weights(model_id="openai-community/gpt2", verbose=False):
    """
    GPT-2のsafetensorsを読み込み、GPT2Paramsに組み立てて返す。
    事前に `make download` を実行して重みファイルを取得してください。
    """
    file_path = f"weights/{model_id}/model.safetensors"

    # 1. 重みをnumpy配列として読み込む
    if verbose:
        print(f"{file_path} からメモリに重みを読み込み中...")
    weights = load_file(file_path)

    # 2. paramsの構造にマッピング
    prefix = "transformer." if "transformer.wte.weight" in weights else ""

    n_layer = sum(1 for key in weights.keys() if key.endswith(".ln_1.weight"))
    if verbose:
        print(f"トランスフォーマーブロック数: {n_layer}")

    blocks = []
    for i in range(n_layer):
        blocks.append(BlockParams(
            ln_1=LayerNormParams(
                g=weights[f"{prefix}h.{i}.ln_1.weight"],
                b=weights[f"{prefix}h.{i}.ln_1.bias"],
            ),
            attn=AttentionParams(
                # GPT-2のConv1D重みは既に(embed_dim, out_dim)形式なので
                # x @ w_qkv をそのまま使える。転置は不要。
                w_qkv=weights[f"{prefix}h.{i}.attn.c_attn.weight"],
                b_qkv=weights[f"{prefix}h.{i}.attn.c_attn.bias"],
                w_out=weights[f"{prefix}h.{i}.attn.c_proj.weight"],
                b_out=weights[f"{prefix}h.{i}.attn.c_proj.bias"],
            ),
            ln_2=LayerNormParams(
                g=weights[f"{prefix}h.{i}.ln_2.weight"],
                b=weights[f"{prefix}h.{i}.ln_2.bias"],
            ),
            mlp=MLPParams(
                w_fc=weights[f"{prefix}h.{i}.mlp.c_fc.weight"],
                b_fc=weights[f"{prefix}h.{i}.mlp.c_fc.bias"],
                w_proj=weights[f"{prefix}h.{i}.mlp.c_proj.weight"],
                b_proj=weights[f"{prefix}h.{i}.mlp.c_proj.bias"],
            ),
        ))

    return GPT2Params(
        wte=weights[f"{prefix}wte.weight"],
        wpe=weights[f"{prefix}wpe.weight"],
        ln_f=LayerNormParams(
            g=weights[f"{prefix}ln_f.weight"],
            b=weights[f"{prefix}ln_f.bias"],
        ),
        blocks=blocks,
    )
