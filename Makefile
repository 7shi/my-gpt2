.PHONY: download run clean help

# Default model ID
MODEL_ID = openai-community/gpt2
WEIGHTS_DIR = weights

# URLs
BASE_URL = https://huggingface.co/$(MODEL_ID)/resolve/main
SAFE_TENSORS_URL = $(BASE_URL)/model.safetensors
VOCAB_URL = $(BASE_URL)/vocab.json
MERGES_URL = $(BASE_URL)/merges.txt

help:
	@echo "Usage:"
	@echo "  make download   - Download GPT-2 weights and tokenizer files"
	@echo "  make run        - Run generation with default prompt"
	@echo "  make clean      - Remove downloaded weights"

download:
	@mkdir -p $(WEIGHTS_DIR)
	@echo "Downloading weights and tokenizer files to $(WEIGHTS_DIR)..."
	@curl -L -o $(WEIGHTS_DIR)/model.safetensors $(SAFE_TENSORS_URL)
	@curl -L -o $(WEIGHTS_DIR)/vocab.json $(VOCAB_URL)
	@curl -L -o $(WEIGHTS_DIR)/merges.txt $(MERGES_URL)
	@echo "Download complete."

run:
	@uv run my_gpt2/generate.py "The quick brown fox" -n 20

clean:
	rm -rf $(WEIGHTS_DIR)
