.PHONY: download download-gpt2 download-rinna run clean help

GPT2_ID  = openai-community/gpt2
RINNA_ID = rinna/japanese-gpt2-small

# 引数: (1)=model_id, (2)=tokenizer_files
define download_model
	@mkdir -p weights/$(1)
	@BASE=https://huggingface.co/$(1)/resolve/main; \
	dl() { f=weights/$(1)/$$1; if [ -f "$$f" ]; then echo "  skip: $$f"; else curl -L -o "$$f" "$$BASE/$$1"; fi; }; \
	dl model.safetensors; \
	dl config.json; \
	for f in $(2); do dl $$f; done
endef

help:
	@echo "Usage:"
	@echo "  make download        - Download both models"
	@echo "  make download-gpt2   - Download openai-community/gpt2"
	@echo "  make download-rinna  - Download rinna/japanese-gpt2-small"
	@echo "  make run             - Run generation with default prompt"
	@echo "  make clean           - Remove all downloaded weights"

download: download-gpt2 download-rinna

download-gpt2:
	$(call download_model,$(GPT2_ID),vocab.json merges.txt)

download-rinna:
	$(call download_model,$(RINNA_ID),spiece.model)

run:
	@uv run my-gpt2 "The quick brown fox" -n 20

clean:
	rm -rf weights
