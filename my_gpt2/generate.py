import numpy as np
import argparse
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.model import GPT2
from my_gpt2.loader import load_gpt2_weights

def generate(prompt, n_tokens_to_generate=30, model_id="openai-community/gpt2"):
    # 1. Load custom tokenizer and weights
    tokenizer = Tokenizer()
    params = load_gpt2_weights(model_id)
    
    # 2. Initialize model
    # GPT-2 small (124M) uses 12 heads
    model = GPT2(params, n_head=12)
    
    # 3. Tokenize input
    input_ids = tokenizer.encode(prompt)
    
    print(f"\nPrompt: '{prompt}'")
    print("Generating: ", end="", flush=True)
    
    # 4. Generation loop
    for _ in range(n_tokens_to_generate):
        # Ensure we don't exceed the model's maximum position embedding
        # GPT-2 default max_pos is 1024
        inputs = np.array([input_ids[-1024:]])
        
        # Forward pass
        logits = model(inputs) # (1, seq_len, vocab_size)
        
        # Get the logits of the last token
        next_token_logits = logits[0, -1, :]
        
        # Greedy search: take the token with max probability
        next_token = int(np.argmax(next_token_logits))
        
        # Append to sequence
        input_ids.append(next_token)
        
        # Print the newly generated token
        print(tokenizer.decode([next_token]), end="", flush=True)
        
        # Stop if we hit the end-of-text token (50256 for GPT-2)
        if next_token == 50256:
            print("\n[End of text reached]")
            break
        
    print("\n\nFull output:")
    return tokenizer.decode(input_ids)

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Scratch Inference with NumPy")
    # Position argument for prompt (one or more words)
    parser.add_argument("prompt", nargs="+", help="Prompt text to start generation")
    parser.add_argument("-n", "--n_tokens", type=int, default=30, help="Number of tokens to generate")
    parser.add_argument("-m", "--model", type=str, default="openai-community/gpt2", help="Model ID or local path")
    
    args = parser.parse_args()
    
    # Join the prompt words with space
    prompt_text = " ".join(args.prompt)
    
    output = generate(prompt_text, n_tokens_to_generate=args.n_tokens, model_id=args.model)
    print(output)

if __name__ == "__main__":
    main()
