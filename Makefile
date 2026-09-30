PYTHON ?= python3
# ids of the eight judges on OpenRouter; for another OpenAI-compatible endpoint: make judges MODELS="..."
MODELS ?= openai/gpt-5.5 anthropic/claude-opus-4.6 google/gemini-3-flash-preview qwen/qwen3.5-397b-a17b openai/gpt-oss-120b google/gemma-3-27b-it openai/gpt-oss-20b deepseek/deepseek-v3.2

.PHONY: install reproduce judges

install:    ## install pinned dependencies (Python 3.12)
	$(PYTHON) -m pip install -r requirements.txt

reproduce:  ## recompute every table of the paper from logs/ -> judges/paper_stats.md, judges/paper_stats.json
	$(PYTHON) code/paper_stats.py

judges:     ## re-run the interleaved EN/RU control design (needs LLM_BASE_URL and LLM_API_KEY)
	$(PYTHON) code/run_judges.py --models $(MODELS) --langs en ru --interleave --seed 20260930 --out-suffix _rerun
