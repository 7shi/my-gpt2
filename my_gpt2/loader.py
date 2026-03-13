import numpy as np
from safetensors.numpy import load_file

def load_gpt2_weights():
    """
    Load GPT-2 safetensors and map to our params dictionary.
    Run `make download` first to get the weights.
    """
    file_path = "weights/model.safetensors"

    # 1. Load weights as numpy arrays
    print(f"Loading weights from {file_path} into memory...")
    weights = load_file(file_path)
    
    # 2. Map to our params structure
    prefix = "transformer." if "transformer.wte.weight" in weights else ""
    
    params = {
        "wte": weights[f"{prefix}wte.weight"],
        "wpe": weights[f"{prefix}wpe.weight"],
        "ln_f": {"g": weights[f"{prefix}ln_f.weight"], "b": weights[f"{prefix}ln_f.bias"]},
        "blocks": []
    }
    
    n_layer = sum(1 for key in weights.keys() if key.endswith(".ln_1.weight"))
    print(f"Detected {n_layer} transformer blocks.")
    
    for i in range(n_layer):
        block_params = {
            "ln_1": {
                "g": weights[f"{prefix}h.{i}.ln_1.weight"],
                "b": weights[f"{prefix}h.{i}.ln_1.bias"]
            },
            "attn": {
                # Note: GPT-2 Conv1D weights are already (embed_dim, out_dim)
                # So we can use x @ w_qkv directly. No transpose needed.
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
