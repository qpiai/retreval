"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Chat, { type ChatMessage } from "@/components/Chat";
import LogPanel from "@/components/LogPanel";
import TreeView from "@/components/TreeView";
import { useSolveStream } from "@/lib/useSolveStream";

const PROVIDERS = ["gemini", "openai", "ollama", "vllm"];

function ThemeToggle() {
  const [dark, setDark] = useState(true);
  useEffect(() => {
    setDark(!document.documentElement.classList.contains("dark") ? false : true);
  }, []);
  const toggle = () => {
    const isDark = document.documentElement.classList.toggle("dark");
    try {
      localStorage.setItem("retreval-theme", isDark ? "dark" : "light");
    } catch {}
    setDark(isDark);
  };
  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className="rounded-lg px-2 py-1 panel text-sm hover:brightness-110"
    >
      {dark ? "🌙" : "☀️"}
    </button>
  );
}

export default function Home() {
  const s = useSolveStream();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [provider, setProvider] = useState("gemini");
  const [iterations, setIterations] = useState(1);
  const [memory, setMemory] = useState(true);

  // Resizable / collapsible split between chat (left) and tree+logs (right).
  const [leftPct, setLeftPct] = useState(50); // default 50/50
  const [rightOpen, setRightOpen] = useState(true);
  const splitRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const onDrag = useCallback((e: MouseEvent) => {
    if (!draggingRef.current || !splitRef.current) return;
    const r = splitRef.current.getBoundingClientRect();
    const pct = ((e.clientX - r.left) / r.width) * 100;
    setLeftPct(Math.min(78, Math.max(22, pct)));
  }, []);

  const startDrag = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      draggingRef.current = true;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      const stop = () => {
        draggingRef.current = false;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        window.removeEventListener("mousemove", onDrag);
        window.removeEventListener("mouseup", stop);
      };
      window.addEventListener("mousemove", onDrag);
      window.addEventListener("mouseup", stop);
    },
    [onDrag]
  );

  // Sync the provider dropdown to whatever the backend is actually configured
  // with (LLM_PROVIDER), so the UI reflects reality instead of a hardcoded default.
  useEffect(() => {
    fetch(`${s.apiBase}/api/health`)
      .then((r) => r.json())
      .then((h) => {
        if (h?.provider) setProvider(h.provider);
      })
      .catch(() => {});
  }, [s.apiBase]);

  const currentStepLabel = s.steps.length
    ? s.steps[s.steps.length - 1].label
    : "";

  const nodeCount = useMemo(
    () => (s.tree.root ? 1 : 0) + Object.keys(s.tree.nodes || {}).length,
    [s.tree]
  );

  const send = (text: string) => {
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", running: true, final: null },
    ]);
    s.start(text, { provider, iterations, memory });
  };

  useEffect(() => {
    if (!s.final) return;
    const content = (
      s.final.answer_text ||
      s.final.best_thought ||
      s.final.predicted_answer ||
      "Done."
    ).trim();
    setMessages((prev) => {
      if (!prev.length) return prev;
      const last = prev[prev.length - 1];
      if (last.role !== "assistant" || !last.running) return prev;
      const copy = [...prev];
      copy[copy.length - 1] = { role: "assistant", content, running: false, final: s.final };
      return copy;
    });
  }, [s.final]);

  useEffect(() => {
    if (s.phase !== "error") return;
    setMessages((prev) => {
      if (!prev.length) return prev;
      const last = prev[prev.length - 1];
      if (!last.running) return prev;
      const copy = [...prev];
      copy[copy.length - 1] = {
        role: "assistant",
        content: `⚠️ ${s.errorMsg || "Something went wrong."}`,
        running: false,
        final: null,
      };
      return copy;
    });
  }, [s.phase, s.errorMsg]);

  const phaseStyle =
    s.phase === "running"
      ? "bg-amber-500/15 text-amber-600 dark:text-amber-300"
      : s.phase === "error"
      ? "bg-red-500/15 text-red-600 dark:text-red-300"
      : s.phase === "done"
      ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300"
      : "bg-black/5 dark:bg-white/5 text-slate-500 dark:text-white/40";

  return (
    <main className="h-screen flex flex-col" data-phase={s.phase}>
      {/* Header */}
      <header className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-black/5 dark:border-white/5">
        <div className="flex items-baseline gap-2.5">
          <span className="text-lg font-extrabold tracking-tight gradient-text">
            ReTreVal
          </span>
          <span className="text-slate-400 dark:text-white/35 text-xs hidden sm:inline">
            Reasoning Tree with Validation
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            className="panel px-2 py-1 text-slate-700 dark:text-white/80 bg-transparent"
          >
            {PROVIDERS.map((p) => (
              <option key={p} value={p} className="bg-white dark:bg-bg-1">
                {p}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1 text-slate-500 dark:text-white/60">
            iters
            <input
              type="number"
              min={1}
              max={5}
              value={iterations}
              onChange={(e) => setIterations(Number(e.target.value))}
              className="w-12 panel px-1.5 py-1 bg-transparent text-slate-700 dark:text-white/80"
            />
          </label>
          <label className="flex items-center gap-1 text-slate-500 dark:text-white/60 cursor-pointer">
            <input
              type="checkbox"
              checked={memory}
              onChange={(e) => setMemory(e.target.checked)}
            />
            memory
          </label>
          <span className={`px-2 py-1 rounded font-mono ${phaseStyle}`} data-testid="phase">
            {s.phase}
          </span>
          <ThemeToggle />
        </div>
      </header>

      {/* Body — resizable, collapsible split */}
      <div ref={splitRef} className="flex-1 flex min-h-0 relative">
        {/* Left: chat */}
        <section
          className="min-w-0 border-r border-black/5 dark:border-white/5"
          style={{ width: rightOpen ? `${leftPct}%` : "100%" }}
        >
          <Chat
            messages={messages}
            phase={s.phase}
            currentStepLabel={currentStepLabel}
            onSend={send}
          />
        </section>

        {/* Drag handle */}
        {rightOpen && (
          <div
            onMouseDown={startDrag}
            onDoubleClick={() => setLeftPct(50)}
            title="Drag to resize · double-click to reset"
            className="group relative w-1.5 shrink-0 cursor-col-resize bg-black/5 dark:bg-white/5 hover:bg-accent2/60 transition-colors"
          >
            <div className="absolute inset-y-0 -left-1.5 -right-1.5" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-8 w-0.5 rounded bg-slate-400/40 group-hover:bg-white/70" />
          </div>
        )}

        {/* Right: tree + logs */}
        {rightOpen && (
          <section
            className="min-w-0 grid grid-rows-[1.45fr_1fr]"
            style={{ width: `${100 - leftPct}%` }}
          >
            <div className="relative min-h-0 border-b border-black/5 dark:border-white/5">
              <div className="absolute z-10 top-2 left-3 right-9 flex flex-wrap gap-3 text-[11px] text-slate-500 dark:text-white/45">
                <span className="uppercase tracking-wider font-medium">
                  Reasoning tree
                </span>
                <span>{nodeCount} nodes</span>
                {s.treeMeta.best_score != null && Number(s.treeMeta.best_score) > 0 && (
                  <span className="text-emerald-600 dark:text-emerald-400">
                    best {Number(s.treeMeta.best_score).toFixed(2)}
                  </span>
                )}
                {s.treeMeta.complexity != null && (
                  <span>complexity {s.treeMeta.complexity}</span>
                )}
              </div>
              <button
                onClick={() => setRightOpen(false)}
                title="Collapse panel"
                aria-label="Collapse panel"
                className="absolute z-10 top-2 right-2 h-6 w-6 grid place-items-center rounded-md panel text-slate-500 dark:text-white/60 hover:text-slate-900 dark:hover:text-white"
              >
                ✕
              </button>
              <TreeView
                tree={s.tree}
                bestNodeId={s.treeMeta.best_node_id}
                currentNodeId={s.treeMeta.current_node_id}
              />
            </div>
            <div className="min-h-0">
              <LogPanel logs={s.logs} steps={s.steps} />
            </div>
          </section>
        )}

        {/* Reveal button when collapsed */}
        {!rightOpen && (
          <button
            onClick={() => setRightOpen(true)}
            title="Show reasoning panel"
            className="absolute z-10 top-3 right-3 panel px-3 py-1.5 text-xs text-slate-600 dark:text-white/70 hover:text-slate-900 dark:hover:text-white flex items-center gap-1.5"
          >
            🌳 Show tree
            {nodeCount > 0 && (
              <span className="text-slate-400 dark:text-white/40">({nodeCount})</span>
            )}
          </button>
        )}
      </div>
    </main>
  );
}
