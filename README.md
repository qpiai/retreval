# ReTreVal

ReTreVal is a reasoning framework for large language models that combines
tree-style exploration, self-refinement, critique-based scoring, and reflexion
memory for multi-step reasoning tasks.

This repository is open-sourced by QPIAI and maintained by Abhishek HS.

## Features

- Adaptive reasoning tree exploration
- Iterative self-refinement of candidate solutions
- Local and critic-based scoring
- Reflexion memory for reusable insights and failure patterns
- Baselines for ReAct, Reflexion, Self-Refine, and Tree-of-Thoughts
- Provider support for Gemini, OpenAI, Ollama, and vLLM-compatible endpoints

## Repository Layout

```text
.
+-- src/
|   +-- agents/       # Reasoning agents, benchmark runners, and prompts
|   +-- clients/      # LLM provider clients
|   +-- tools/        # Planning, scoring, search, synthesis, and computation tools
|   +-- utils/        # Memory and KV-cache utilities
+-- data/             # Example benchmark inputs
+-- requirements.txt
+-- .env.example
+-- analyze_results.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Then set the API key for the provider you want to use.

## Configuration

Choose one provider with `LLM_PROVIDER`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## Usage

Run a Math500 benchmark with a vLLM-compatible endpoint:

```bash
python run_math500_vllm.py
```

Run the KV-cache variant:

```bash
python run_math500_vllm_kvcache_10.py
```

Run an agent script directly:

```bash
python -m src.agents.agent_main --prompt "Solve a multi-step reasoning problem" --iters 5
```

Analyze HumanEval result CSVs:

```bash
python analyze_results.py --results-dir results/humaneval
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Authors

See [AUTHORS.md](AUTHORS.md).
