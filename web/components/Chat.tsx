"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Markdown from "./Markdown";
import type { FinalEvent, Phase } from "@/lib/types";

/** Render a short answer string as inline math when it contains LaTeX, else plain. */
function answerMarkdown(ans: string): string {
  if (!ans) return "—";
  const stripped = ans.replace(/^\\boxed\{([\s\S]+)\}$/, "$1").trim();
  const hasTex = /[\\^_{}]/.test(stripped);
  return hasTex ? `$${stripped}$` : stripped;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  final?: FinalEvent | null;
  running?: boolean;
}

const EXAMPLES = [
  // math
  "How many positive whole-number divisors does 196 have?",
  "If 3x + 7 = 22, what is x?",
  "A bag has 4 red and 6 blue balls. P(two reds, no replacement)?",
  // other domains — harder, to show the agent is general
  "Implement Dijkstra's shortest-path algorithm in Python and run it on a small example graph.",
  "What is the de Broglie wavelength of an electron moving at 2.0×10⁶ m/s? Show the formula and the numeric value.",
];

export default function Chat({
  messages,
  phase,
  currentStepLabel,
  onSend,
}: {
  messages: ChatMessage[];
  phase: Phase;
  currentStepLabel?: string;
  onSend: (text: string) => void;
}) {
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = phase === "running";

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStepLabel]);

  const submit = () => {
    const v = inputRef.current?.value.trim();
    if (!v || busy) return;
    onSend(v);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <div className="mt-10 text-center">
            <div className="text-slate-400 dark:text-white/35 text-sm">
              Ask ReTreVal a problem — it explores a tree of approaches, verifies
              with tools, and keeps the best.
            </div>
            <div className="mt-4 flex flex-col items-center gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  data-testid="example-chip"
                  onClick={() => onSend(ex)}
                  className="chip max-w-full truncate"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
            >
              <div
                className={
                  "max-w-[88%] rounded-2xl px-3.5 py-2.5 text-sm " +
                  (m.role === "user"
                    ? "bg-gradient-to-br from-accent2 to-accent text-white rounded-br-sm shadow-lg shadow-accent2/20"
                    : "panel rounded-bl-sm text-slate-800 dark:text-white/90")
                }
              >
                {m.running && !m.content ? (
                  <span className="flex items-center gap-2 text-slate-500 dark:text-white/55 italic">
                    <span className="inline-block h-2 w-2 rounded-full bg-accent2 animate-pulseGlow" />
                    {currentStepLabel ? `${currentStepLabel}…` : "Thinking…"}
                  </span>
                ) : m.role === "assistant" ? (
                  <Markdown>{m.content}</Markdown>
                ) : (
                  <span className="whitespace-pre-wrap">{m.content}</span>
                )}

                {m.final && (
                  <div className="mt-2 pt-2 border-t border-black/10 dark:border-white/10 text-xs text-slate-500 dark:text-white/50 space-y-1">
                    <div className="flex items-baseline gap-1">
                      <span className="opacity-70">answer:</span>
                      <span className="text-emerald-600 dark:text-emerald-300">
                        <Markdown inline className="font-mono">
                          {answerMarkdown(m.final.predicted_answer)}
                        </Markdown>
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5">
                      <span>score {(m.final.score ?? 0).toFixed(2)}</span>
                      {m.final.complexity != null && (
                        <span>complexity {m.final.complexity}</span>
                      )}
                      {m.final.total_backtracks > 0 && (
                        <span>{m.final.total_backtracks} backtrack(s)</span>
                      )}
                      {m.final.is_decomposed && <span>decomposed</span>}
                    </div>
                    {m.final.tools_used?.length > 0 && (
                      <div className="flex flex-wrap gap-1 pt-0.5">
                        {m.final.tools_used.map((t) => (
                          <span
                            key={t}
                            className="px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 font-mono text-[10px]"
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={endRef} />
      </div>

      <div className="border-t border-black/5 dark:border-white/5 p-3">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            rows={1}
            data-testid="chat-input"
            placeholder={busy ? "Agent is reasoning…" : "Ask a problem…"}
            disabled={busy}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            className="flex-1 resize-none rounded-xl panel px-3 py-2.5 text-sm text-slate-800 dark:text-white/90 placeholder:text-slate-400 dark:placeholder:text-white/30 focus:outline-none focus:ring-2 focus:ring-accent2/50 disabled:opacity-50"
          />
          <button
            onClick={submit}
            disabled={busy}
            data-testid="send-btn"
            className="rounded-xl bg-gradient-to-br from-accent to-accent2 px-4 py-2.5 text-sm font-medium text-white disabled:opacity-40 hover:brightness-110 transition shadow-lg shadow-accent/20"
          >
            {busy ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
