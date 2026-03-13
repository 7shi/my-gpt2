import numpy as np
import argparse
from .tokenizer import Tokenizer
from .model import GPT2
from .loader import load_gpt2_weights
from .model import softmax

def generate(prompt, n_tokens_to_generate=30, temperature=1.0):
    # 1. Load custom tokenizer and weights
    tokenizer = Tokenizer()
    params = load_gpt2_weights()
    
    # 2. Initialize model
    # GPT-2 small (124M) uses 12 heads
    model = GPT2(params, n_head=12)
    
    # 3. Tokenize input
    input_ids = tokenizer.encode(prompt)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Temperature: {temperature}")
    print("Generating: ", end="", flush=True)
    
    # 4. Generation loop
    # Buffer for incomplete multi-byte characters
    byte_buffer = bytearray()
    
    for _ in range(n_tokens_to_generate):
        # Ensure we don't exceed the model's maximum position embedding
        inputs = np.array([input_ids[-1024:]])
        
        # Forward pass
        logits = model(inputs)
        
        # Get the logits of the last token
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        if temperature > 0:
            # Scale logits and use random sampling
            next_token_logits = next_token_logits / temperature
            probs = softmax(next_token_logits)
            next_token = int(np.random.choice(len(probs), p=probs))
        else:
            # Greedy search if temperature is 0
            next_token = int(np.argmax(next_token_logits))
        
        # Append to sequence
        input_ids.append(next_token)
        
        # Get raw bytes of the new token
        token_str = tokenizer.decoder[next_token]
        token_bytes = bytes([tokenizer.byte_decoder[c] for c in token_str])
        byte_buffer.extend(token_bytes)
        
        # Try to decode the buffer as UTF-8
        try:
            # If the whole buffer is valid UTF-8
            decoded_text = byte_buffer.decode("utf-8")
            print(decoded_text, end="", flush=True)
            byte_buffer.clear()
        except UnicodeDecodeError as e:
            # Decode only the valid part
            valid_bytes = byte_buffer[:e.start]
            if valid_bytes:
                print(valid_bytes.decode("utf-8"), end="", flush=True)
                # Keep the invalid part for the next token
                del byte_buffer[:e.start]
        
        # Stop if we hit the end-of-text token (50256)
        if next_token == 50256:
            print("\n[End of text reached]")
            break
        
    print("\n\nFull output:")
    # Final decode: ignore any incomplete multi-byte characters at the end
    full_bytes = bytearray()
    for tid in input_ids:
        token_str = tokenizer.decoder[tid]
        full_bytes.extend([tokenizer.byte_decoder[c] for c in token_str])
    
    return full_bytes.decode("utf-8", errors="ignore")

def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 Scratch Inference with NumPy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Position argument for prompt (one or more words)
    parser.add_argument("prompt", nargs="+", help="Prompt text to start generation")
    parser.add_argument("-n", "--n_tokens", type=int, default=30, help="Number of tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sampling temperature (lower is more deterministic)")
    
    args = parser.parse_args()
    
    # Join the prompt words with space
    prompt_text = " ".join(args.prompt)
    
    output = generate(prompt_text, n_tokens_to_generate=args.n_tokens, temperature=args.temperature)
    print(output)

if __name__ == "__main__":
    main()
