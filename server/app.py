"""FastAPI bridge that streams the ReTreVal agent's reasoning to the web UI.

The agent (``src.agents.agent_langgraph.LangGraphAgent``) is a compiled LangGraph
state graph. We drive it with ``app.stream(..., stream_mode=["updates","values"])``
so we can emit, after every super-step:

  * ``log``   — raw stdout the tools printed during that step
  * ``step``  — which node ran + key scalars (current/best node, score, complexity)
  * ``tree``  — the full reasoning-tree snapshot ({root, nodes}) so the UI can redraw
  * ``final`` — the finished result (predicted answer, best thought, tools used)
  * ``error`` — any exception, surfaced to the client

Set ``RETREVAL_MOCK=1`` to stream a canned run with no LLM/key required — handy for
developing the UI. Drop a real key into ``.env`` (GEMINI_API_KEY=...) and unset the
flag to run the real agent.

Run:
    cd retreval_oss
    . .venv/bin/activate
    uvicorn server.app:app --reload --port 8000
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Make the repo root importable (so `src...` resolves) regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

app = FastAPI(title="ReTreVal UI bridge")

# Local dev/demo bridge — the UI may be served from any localhost port
# (Next dev on :7575, the static export, etc.), so allow all origins. No
# credentials are sent, so the wildcard is safe here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# SSE helpers
# --------------------------------------------------------------------------- #
def _sse(event: str, data: Any) -> str:
    """Format one Server-Sent Event frame."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _clean_answer(text: str) -> str:
    """Tidy the synthesized answer for display: trim, drop blank-line runs, and
    collapse consecutive duplicate lines (models sometimes echo the answer twice)."""
    if not text:
        return ""
    out: list = []
    prev = None
    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if line.strip() and line.strip() == prev:
            continue  # drop an immediately repeated line
        if not line.strip() and (not out or not out[-1].strip()):
            continue  # collapse blank runs
        out.append(line)
        if line.strip():
            prev = line.strip()
    return "\n".join(out).strip()


def _node_label(node_name: str) -> str:
    return {
        "plan": "🗺️  Plan",
        "expand_tree": "🌳 Expand tree",
        "react_refine": "🛠️  Tool-augmented refine",
        "critique": "🔍 Critique",
        "score": "📊 Dual score",
        "memory": "🧠 Memory update",
        "complexity": "📐 Complexity",
        "synthesize": "🧩 Synthesize",
        "validate": "✅ Validate",
        "feedback_control": "🔁 Feedback control",
        "backtrack": "↩️  Backtrack",
        "backtrack_refine": "↩️  Backtrack refine",
        "analyze_failure": "🔬 Analyze failure",
        "decompose": "🪓 Decompose",
        "run_subtasks": "🏃 Run sub-tasks",
    }.get(node_name, node_name)


# --------------------------------------------------------------------------- #
# Real agent run — produced on a worker thread, consumed as SSE on the request
# --------------------------------------------------------------------------- #
def _run_agent_to_queue(params: Dict[str, Any], q: "queue.Queue[Optional[str]]") -> None:
    """Drive the LangGraph agent and push SSE frames into ``q``. None == done."""
    try:
        from src.agents.agent_langgraph import LangGraphAgent, extract_answer

        agent = LangGraphAgent(
            max_loops=int(params.get("iterations", 2)),
            verbose=True,
            provider=params.get("provider") or None,
            model=params.get("model") or None,
            use_memory=bool(params.get("memory", True)),
            oracle=False,
        )

        problem = params["problem"]
        agent.cache_mgr.set_problem_context(problem)

        # Mirror LangGraphAgent.run()'s initial state (kept in sync with the agent).
        state: Dict[str, Any] = {
            "problem": problem,
            "critique_history": [],
            "iterations": 0,
            "max_depth": int(os.getenv("MAX_DEPTH", "5")),
            "children_per_expansion": int(os.getenv("CHILDREN_PER_EXPANSION", "3")),
            "problem_id": "",
            "expected_answer": "",
            "is_correct": False,
            "predicted_answer": "",
            "error_analysis": "",
            "failure_type": "",
            "approach_type": "",
            "feedback_loops": 0,
            "should_retry": False,
            "failed_nodes": [],
            "backtrack_history": [],
            "failure_contexts": [],
            "explored_root_children": [],
            "backtrack_available": False,
            "backtracking_iteration": 0,
            "subtasks": [],
            "subtask_results": {},
            "is_decomposed": False,
            "is_subtask": False,
            "aggregated_result": "",
            "react_traces": [],
            "tools_used": [],
        }
        default_limit = 100 + 12 * int(agent.max_loops)
        recursion_limit = int(os.getenv("RECURSION_LIMIT", str(default_limit)))

        q.put(_sse("status", {"phase": "running", "provider": agent.llm.provider}))

        gen = agent.app.stream(
            state,
            stream_mode=["updates", "values"],
            config={"recursion_limit": recursion_limit},
        )
        buf = io.StringIO()
        last_state: Dict[str, Any] = dict(state)

        while True:
            buf.seek(0)
            buf.truncate(0)
            with contextlib.redirect_stdout(buf):
                try:
                    mode, chunk = next(gen)
                except StopIteration:
                    break
                except Exception as exc:  # graph error mid-run
                    q.put(_sse("error", {"message": str(exc)}))
                    break

            logs = buf.getvalue()
            if logs.strip():
                for line in logs.splitlines():
                    if line.strip():
                        q.put(_sse("log", {"line": line}))

            if mode == "updates" and isinstance(chunk, dict):
                node_name = next(iter(chunk.keys()), "")
                q.put(_sse("step", {"node": node_name, "label": _node_label(node_name)}))
            elif mode == "values" and isinstance(chunk, dict):
                last_state = chunk
                tree = chunk.get("tree") or {}
                q.put(
                    _sse(
                        "tree",
                        {
                            "tree": tree,
                            "current_node_id": chunk.get("current_node_id"),
                            "best_node_id": chunk.get("best_node_id"),
                            "best_score": chunk.get("best_score"),
                            "complexity": chunk.get("complexity"),
                            "iterations": chunk.get("iterations"),
                        },
                    )
                )

        if agent.use_memory:
            with contextlib.redirect_stdout(buf):
                agent.memory.save()

        predicted = last_state.get("predicted_answer") or extract_answer(
            last_state.get("final_output", "") or last_state.get("best_thought", "")
        )
        # The chat bubble shows the synthesized final answer (final_output) — NOT
        # the raw best-node thought, which is often just a heading or a fragment.
        answer_text = _clean_answer(
            last_state.get("final_output", "")
            or last_state.get("aggregated_result", "")
            or last_state.get("best_thought", "")
            or last_state.get("current_thought", "")
            or predicted
        )
        q.put(
            _sse(
                "final",
                {
                    "predicted_answer": predicted,
                    "answer_text": answer_text,
                    "best_thought": last_state.get("best_thought") or "",
                    "plan": last_state.get("plan", ""),
                    "score": last_state.get("best_score", last_state.get("score", 0.0)),
                    "tools_used": sorted(set(last_state.get("tools_used", []))),
                    "complexity": last_state.get("complexity"),
                    "is_decomposed": last_state.get("is_decomposed", False),
                    "total_backtracks": len(last_state.get("backtrack_history", [])),
                },
            )
        )
    except Exception as exc:  # setup/import/provider error
        q.put(_sse("error", {"message": f"{type(exc).__name__}: {exc}"}))
    finally:
        q.put(None)


def _agent_event_stream(params: Dict[str, Any]) -> Iterator[str]:
    q: "queue.Queue[Optional[str]]" = queue.Queue()
    worker = threading.Thread(target=_run_agent_to_queue, args=(params, q), daemon=True)
    worker.start()
    while True:
        frame = q.get()
        if frame is None:
            break
        yield frame
    yield _sse("done", {})


# --------------------------------------------------------------------------- #
# Mock run — no LLM/key required, lets the UI be developed offline
# --------------------------------------------------------------------------- #
def _mock_narrative(problem: str) -> Dict[str, Any]:
    """Pick a deterministic, domain-distinct narrative from the prompt text.

    Two narratives ship so the demo can show the agent on two different kinds of
    task (number theory + a deep-learning knowledge question) back-to-back.
    """
    p = (problem or "").lower()

    # --- Narrative B: CS / deep-learning knowledge question (uses lookup tools) ---
    if any(k in p for k in ("gelu", "activation", "who introduced", "who proposed", "formula")):
        return {
            "complexity": 2,
            "plan": "Identify the activation function, recall its origin, and verify the exact formula.",
            "tools": ["arxiv_search", "wikipedia"],
            "predicted": "GELU(x) = x·Φ(x)",
            "answer_text": (
                "**GELU** (Gaussian Error Linear Unit) was introduced by "
                "**Hendrycks & Gimpel (2016)**.\n\n"
                "$\\mathrm{GELU}(x) = x\\,\\Phi(x) = 0.5\\,x\\left(1 + "
                "\\operatorname{erf}\\!\\left(\\tfrac{x}{\\sqrt{2}}\\right)\\right)$\n\n"
                "where $\\Phi$ is the standard-normal CDF. Verified via `arxiv_search` "
                "and `wikipedia`."
            ),
            "script": [
                ("plan", [("log", "🗺️  PLANNING — factual / deep-learning question"),
                          ("log", "   complexity=2 → recall, then verify against a source")]),
                ("expand_tree", [("log", "🔎 EXPANSION — proposing 3 approaches"),
                                 ("log", "  A. Recall GELU from training knowledge"),
                                 ("log", "  B. Verify against the original arXiv paper"),
                                 ("log", "  C. Assume it's the same as SiLU/Swish"),
                                 ("addchild", ("A", "Recall GELU: introduced by Hendrycks & Gimpel", "root")),
                                 ("addchild", ("B", "Look up the original paper to confirm the formula", "root")),
                                 ("addchild", ("C", "Treat GELU as identical to SiLU / Swish", "root"))]),
                ("react_refine", [("cur", "A"),
                                  ("log", "🤖 REACT REFINE @A — LLM selects tools"),
                                  ("log", "   Action: arxiv_search(\"GELU Gaussian Error Linear Units\")"),
                                  ("log", "   Observation: Hendrycks & Gimpel, 2016 (arXiv:1606.08415)")]),
                ("critique", [("log", "🧪 CRITIQUE — cross-critique A vs B vs C")]),
                ("score", [("log", "📊 SCORE — A 0.83 · B 0.80 · C 0.28"),
                           ("setscore", ("A", 0.83)), ("setscore", ("B", 0.80)),
                           ("setscore", ("C", 0.28))]),
                ("memory", [("log", "🧠 MEMORY — insight: \"verify named methods against the source paper\"")]),
                ("backtrack", [("cur", "C"),
                               ("log", "↩️  BACKTRACK — C scored 0.28 (< 0.4), abandoning branch"),
                               ("log", "   failure: GELU ≠ SiLU — different definition")]),
                ("react_refine", [("cur", "A"),
                                  ("log", "🤖 REACT REFINE @A (depth 2) — confirm the exact formula"),
                                  ("addchild", ("A1", "GELU(x) = x·Φ(x) = 0.5x(1+erf(x/√2))", "A")),
                                  ("log", "   Action: wikipedia(\"Gaussian Error Linear Unit\")"),
                                  ("log", "   Observation: GELU(x) = x·Φ(x)")]),
                ("score", [("cur", "A1"), ("setscore", ("A1", 0.94)),
                           ("log", "📊 SCORE — A1: 0.94  ★ new best (path root→A→A1)")]),
                ("synthesize", [("log", "🧩 SYNTHESIZE — composing the answer + formula")]),
                ("validate", [("log", "✅ VALIDATE — sourced and well-formed, confidence 0.94")]),
            ],
        }

    # --- Narrative A: number theory (default) — uses computation tools ---
    return {
        "complexity": 3,
        "plan": "Estimate complexity, expand 3 approaches, refine + score, verify the winner.",
        "tools": ["python_exec", "calculator"],
        "predicted": "9",
        "answer_text": (
            "**196 = 2² · 7²**, so the number of positive divisors is "
            "**(2+1)(2+1) = 9**.\n\n"
            "Verified with `python_exec` (`sympy.factorint`) and `calculator`."
        ),
        "script": [
            ("plan", [("log", "🗺️  PLANNING — estimating complexity"),
                      ("log", "   complexity=3 → tree depth=2, branching=3")]),
            ("expand_tree", [("log", "🔎 EXPANSION — proposing 3 distinct approaches"),
                             ("log", "  A. Prime-factorize, then count divisors"),
                             ("log", "  B. Enumerate divisors by trial division"),
                             ("log", "  C. Guess from a similar solved problem"),
                             ("addchild", ("A", "Prime-factorize 196, then apply the divisor-count formula", "root")),
                             ("addchild", ("B", "Enumerate divisors by trial division up to √196", "root")),
                             ("addchild", ("C", "Pattern-match against a previously solved problem", "root"))]),
            ("react_refine", [("cur", "A"),
                              ("log", "🤖 REACT REFINE @A — LLM selects tools"),
                              ("log", "   Action: python_exec(\"import sympy; print(sympy.factorint(196))\")"),
                              ("log", "   Observation: {2: 2, 7: 2}")]),
            ("critique", [("log", "🧪 CRITIQUE — cross-critique A vs B vs C")]),
            ("score", [("log", "📊 SCORE — A 0.84 · B 0.58 · C 0.31"),
                       ("setscore", ("A", 0.84)), ("setscore", ("B", 0.58)),
                       ("setscore", ("C", 0.31))]),
            ("memory", [("log", "🧠 MEMORY — insight: \"factor first, then (e₁+1)(e₂+1)\"")]),
            ("backtrack", [("cur", "C"),
                           ("log", "↩️  BACKTRACK — C scored 0.31 (< 0.4), abandoning branch"),
                           ("log", "   failure: pattern-match had no matching exemplar")]),
            ("react_refine", [("cur", "A"),
                              ("log", "🤖 REACT REFINE @A (depth 2) — deepen the winner"),
                              ("addchild", ("A1", "Apply formula: (2+1)(2+1) = 9", "A")),
                              ("log", "   Action: calculator(\"(2+1)*(2+1)\")"),
                              ("log", "   Observation: 9")]),
            ("score", [("cur", "A1"), ("setscore", ("A1", 0.93)),
                       ("log", "📊 SCORE — A1: 0.93  ★ new best (path root→A→A1)")]),
            ("synthesize", [("log", "🧩 SYNTHESIZE — composing final answer from best path")]),
            ("validate", [("log", "✅ VALIDATE — answer well-formed, confidence 0.93")]),
        ],
    }


def _mock_event_stream(params: Dict[str, Any]) -> Iterator[str]:
    problem = params.get("problem", "")
    delay = float(os.getenv("RETREVAL_MOCK_DELAY", "0.35"))
    nar = _mock_narrative(problem)

    def node(d: int, score: float, thought: str, pid: Optional[str], nid: str) -> Dict[str, Any]:
        return {
            "id": nid, "parent_id": pid, "depth": d, "thought": thought,
            "local_score": round(min(1.0, score + 0.1), 2), "cross_score": round(score, 2),
            "combined_score": round(score, 2), "children": [],
            "state": {"type": "idea"}, "refinement_count": 1,
        }

    root = {
        "id": "root", "parent_id": None, "depth": 0,
        "thought": f"Plan: {nar['plan']}", "combined_score": 0.6,
        "local_score": 1.0, "cross_score": 0.0, "children": [],
        "state": {"type": "problem"},
    }
    tree: Dict[str, Any] = {"root": root, "nodes": {}}

    yield _sse("status", {"phase": "running", "provider": "mock"})
    best_node, best_score, cur = None, 0.0, None

    for node_name, ops in nar["script"]:
        time.sleep(delay)
        yield _sse("step", {"node": node_name, "label": _node_label(node_name)})
        for kind, val in ops:
            if kind == "log":
                yield _sse("log", {"line": val})
            elif kind == "cur":
                cur = val
            elif kind == "addchild":
                nid, th, pid = val
                parent = root if pid == "root" else tree["nodes"][pid]
                tree["nodes"][nid] = node(parent["depth"] + 1, 0.0, th, pid, nid)
                parent.setdefault("children", [])
                if nid not in parent["children"]:
                    parent["children"].append(nid)
                cur = nid
            elif kind == "setscore":
                nid, sc = val
                if nid in tree["nodes"]:
                    tree["nodes"][nid]["combined_score"] = sc
                    tree["nodes"][nid]["local_score"] = round(min(1.0, sc + 0.05), 2)
                    tree["nodes"][nid]["cross_score"] = round(max(0.0, sc - 0.05), 2)
                    if sc > best_score:
                        best_score, best_node = sc, nid
            time.sleep(delay * 0.25)
        yield _sse(
            "tree",
            {"tree": tree, "current_node_id": cur, "best_node_id": best_node,
             "best_score": best_score, "complexity": nar["complexity"], "iterations": 1},
        )

    time.sleep(delay)
    yield _sse(
        "final",
        {
            "predicted_answer": nar["predicted"],
            "answer_text": nar["answer_text"],
            "plan": nar["plan"],
            "score": best_score, "tools_used": nar["tools"],
            "complexity": nar["complexity"], "is_decomposed": False, "total_backtracks": 1,
        },
    )
    yield _sse("done", {})


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "mock": os.getenv("RETREVAL_MOCK", "") in ("1", "true", "yes"),
        "provider": os.getenv("LLM_PROVIDER", "gemini"),
    }


@app.get("/api/solve")
def solve(
    problem: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    iterations: int = 2,
    memory: bool = True,
):
    """SSE endpoint. EventSource(GET) friendly — params arrive as query string."""
    params = {
        "problem": problem,
        "provider": provider,
        "model": model,
        "iterations": iterations,
        "memory": memory,
    }
    mock = os.getenv("RETREVAL_MOCK", "") in ("1", "true", "yes")
    stream = _mock_event_stream(params) if mock else _agent_event_stream(params)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
