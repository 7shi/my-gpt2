import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.model import GPT2
from my_gpt2.loader import load_gpt2_weights

def generate(prompt, n_tokens_to_generate=10, model_id="openai-community/gpt2"):
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
        
    print("\n\nFull output:")
    return tokenizer.decode(input_ids)

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Alan Turing was a"
    output = generate(prompt, n_tokens_to_generate=30)
    print(output)
