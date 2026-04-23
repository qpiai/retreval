# ReTreVal

<div align="center">

Reasoning, refinement, and evaluation for math-focused LLM workflows.

[![GitHub](https://img.shields.io/badge/GitHub-ReTreVal-111827?style=flat-square)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-0f766e?style=flat-square)](LICENSE)
[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-7c3aed?style=flat-square)](https://langchain-ai.github.io/langgraph/)

</div>

ReTreVal is a compact open-source reasoning agent for large language models. It keeps the public surface intentionally small: one agent, one Math500 evaluator, and a clean path from prompt to answer.

## ✨ What It Does

- Drafts an answer with LangGraph
- Refines the draft once or more before finalizing
- Extracts the final answer for automated evaluation
- Runs directly on the bundled Math500 CSV
- Supports Gemini, OpenAI, Ollama, and vLLM-compatible backends

## 🧠 Architecture

<div align="center">

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'22px','fontFamily':'Inter, Helvetica, Arial, sans-serif','lineColor':'#64748b'}}}%%
graph LR
    A["👤 Your Problem"] --> B["🤖 ReTreVal Agent"]
    B --> C["🧭 Plan"]
    C --> D["✍️ Draft"]
    D --> E["🪄 Refine"]
    E --> F["🏁 Finalize"]
    F --> G["📦 Answer"]
    G --> H["📊 Math500 Eval"]
    H --> I["📝 JSONL Results"]

    classDef pink fill:#E84393,stroke:#E84393,color:#ffffff,stroke-width:3px,font-size:22px
    classDef blue fill:#1e3a8a,stroke:#1e3a8a,color:#ffffff,stroke-width:3px,font-size:22px

    class B,D,E,F,H,I pink
    class A,C,G blue
```

</div>

## 🚀 Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set one provider in `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

Run the agent on one problem:

```bash
python -m src.agents.agent_main \
  --provider gemini \
  --prompt "How many positive whole-number divisors does 196 have?"
```

Run Math500 evaluation:

```bash
python -m src.agents.math500_eval --provider gemini --limit 5
```

Compatibility wrappers:

```bash
python run_math500_vllm.py --provider gemini --limit 5
python run_math500_vllm_kvcache_10.py --provider gemini --limit 5
```

## 🧩 Repository Layout

```text
.
├── src/
│   ├── agents/     # Agent + Math500 evaluation entry points
│   └── clients/    # LLM provider clients
├── data/           # Bundled Math500 sample data
├── results/        # Evaluation outputs
├── .env.example
├── requirements.txt
└── analyze_results.py
```

## 🔧 Configuration

```env
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash

OPENAI_MODEL=gpt-4o-mini
OLLAMA_MODEL=qwen2.5:7b
VLLM_BASE_URL=http://localhost:8000/v1
```

## 🤝 Acknowledgments

Thanks to [LangGraph](https://langchain-ai.github.io/langgraph/), [Hugging Face](https://huggingface.co/), and the [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) authors, plus the model and tooling communities behind [Gemini](https://ai.google.dev/gemini-api/docs), [OpenAI](https://platform.openai.com/docs), [Ollama](https://ollama.com/), and [vLLM](https://docs.vllm.ai/).

Subtle project credit lives in [AUTHORS.md](AUTHORS.md).

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE).
