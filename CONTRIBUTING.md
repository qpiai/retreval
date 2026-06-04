# 🤝 Contributing to ReTreVal

**Thank you for being here — you're about to help make LLM reasoning better for everyone.**

ReTreVal is built by researchers and engineers who believe reasoning agents should be open, auditable, and continuously improvable. Every contribution — a new task evaluator, a memory strategy, a bug fix, a typo — helps move that forward.

---

## 💡 Why contribute?

- 🧠 **Real research impact** — ReTreVal is an active research framework cited in [arXiv 2601.02880](https://arxiv.org/pdf/2601.02880). Your work can directly influence future experiments
- 🔌 **Broad task coverage** — the agent is task-agnostic; adding new task evaluators (e.g. GPQA, ARC) is one of the most impactful things you can do
- 🤖 **Edge-of-field work** — reflexion memory, tree-structured reasoning, KV cache optimization, multi-backend inference
- 👥 **Responsive maintainers** — we review PRs promptly and are happy to help first-timers

---

## 🌱 Good first contributions

If you're new, any of these are a great start:

- 🐛 Fix a bug from [Issues](../../issues) — look for `good first issue`
- 📝 Improve the docs — a missing step, a confusing example, a typo
- 🧪 Add a test for an untested code path
- 🔌 Add a new task evaluator (GPQA, ARC)
- 🧩 Add a new LLM backend client

Tiny PRs are welcome. A one-line doc fix counts.

---

## 🛠️ Development Setup

```bash
# Clone and create a virtual environment
git clone https://github.com/your-org/retreval.git && cd retreval
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure a provider
cp .env.example .env
# Edit .env: set LLM_PROVIDER and your API key
```

Needs: Python 3.9+

---

## 🧪 Running the Agent

**Single problem:**
```bash
python -m src.agents.agent_main \
  --provider gemini \
  --prompt "Solve: if 2x + 3 = 11, find x." \
  --iterations 2 \
  --verbose
```

**MATH-500 batch evaluation:**
```bash
python -m src.agents.math500_eval --provider gemini --limit 20 --iterations 1
```

**Analyze results:**
```bash
python analyze_results.py --results-dir results/math500
```

If you're adding a new task evaluator, run it end-to-end on at least 10 problems before submitting.

---

## 🔌 Adding a New Task Evaluator

The agent is task-agnostic. To plug in a new benchmark:

1. **Create a loader** in `src/agents/` that yields `(question, expected_answer)` pairs
2. **Create an answer normalizer** — strips formatting, lowercases, handles domain-specific representations
3. **Wire up a runner** following the pattern in `math500_eval.py`
4. **Add sample data** (even 10–20 examples) under `data/your_task/`

No changes to the agent, memory system, or LLM clients are needed.

---

## 🧩 Adding a New LLM Backend

All backends live in `src/clients/` and implement the same interface as `llm_client.py`. To add a new one:

1. Create `src/clients/your_provider_client.py`
2. Implement `_call_api()`, `expand()`, `self_refine()`, and `score_node()`
3. Register it in `llm_client.py`'s provider router
4. Add the relevant env vars to `.env.example`

---

## 🎨 Code Style

Follow what the surrounding file already does. Beyond that:

- **Python** — PEP 8; type hints on public functions
- **No comments** explaining *what* — only *why* when the reason is non-obvious
- **No dead code** — remove unused imports, variables, and fallback paths
- **Tests** — if you're fixing a bug, add a case that catches it

---

## 📬 Pull Request Process

1. 🍴 Fork the repo and create a focused feature branch
2. ✏️ Make your changes; keep commits small and named clearly
3. ✅ Run the affected eval script end-to-end to verify correctness
4. 🚀 Open a merge request with:
   - What changed and why
   - Linked issue number if applicable
   - Sample output for new evaluators or backends
5. 💬 Respond to review — we'll be constructive; we hope you will be too

For bigger ideas (new memory architectures, multi-agent extensions), please open a [Discussion](../../issues) first so we can align early and avoid throwaway work.

---

## 💬 Questions?

- 🐞 Bug reports → [Issues](../../issues)
- 💡 Big-idea proposals → open an Issue with the `proposal` label before coding
- 📄 Paper questions → [arXiv 2601.02880](https://arxiv.org/pdf/2601.02880)

---

## 📜 License

By contributing, you agree that your work is released under the [Apache License 2.0](LICENSE) — the same license as the rest of ReTreVal.

---

Thanks for reading all the way through. Let's build better reasoning agents together.
