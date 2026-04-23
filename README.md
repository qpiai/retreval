# ReTreVal

ReTreVal is a compact reasoning agent for large language models. The public
interface focuses on one agent and one Math500 evaluation path.

This repository is open-sourced by QPIAI and maintained by Abhishek HS.

## Features

- LangGraph-based draft, refine, and answer-extraction flow
- Simple CLI for solving one problem
- Math500 evaluator for the bundled CSV
- Provider support for Gemini, OpenAI, Ollama, and vLLM-compatible endpoints

## Repository Layout

```text
.
+-- src/
|   +-- agents/       # Reasoning agents, benchmark runners, and prompts
|   +-- clients/      # LLM provider clients
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

Run the agent on one problem:

```bash
python -m src.agents.agent_main \
  --provider gemini \
  --prompt "How many positive whole-number divisors does 196 have?"
```

Evaluate the first 5 Math500 examples:

```bash
python -m src.agents.math500_eval --provider gemini --limit 5
```

The compatibility wrappers call the same evaluator:

```bash
python run_math500_vllm.py --provider gemini --limit 5
```

Analyze HumanEval result CSVs:

```bash
python analyze_results.py --results-dir results/humaneval
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Authors

See [AUTHORS.md](AUTHORS.md).
