"use client";

import { useEffect, useRef } from "react";
import type { LogLine, StepEvent } from "@/lib/types";

function lineColor(text: string): string {
  if (/error|fail|❌|⚠️/i.test(text)) return "text-red-500 dark:text-red-400";
  if (/✅|correct|\bbest\b/i.test(text)) return "text-emerald-600 dark:text-emerald-400";
  if (/🔎|expand|🌳/i.test(text)) return "text-sky-600 dark:text-sky-400";
  if (/🤖|react|tool|action|observation/i.test(text))
    return "text-violet-600 dark:text-violet-400";
  if (/🧠|memory|insight/i.test(text)) return "text-pink-600 dark:text-pink-400";
  if (/🗺️|plan/i.test(text)) return "text-amber-600 dark:text-amber-400";
  if (/📊|score/i.test(text)) return "text-cyan-600 dark:text-cyan-400";
  return "text-slate-500 dark:text-white/55";
}

export default function LogPanel({
  logs,
  steps,
}: {
  logs: LogLine[];
  steps: StepEvent[];
}) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [logs.length]);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-black/5 dark:border-white/5 text-[11px] flex-wrap shrink-0">
        <span className="uppercase tracking-wider text-slate-400 dark:text-white/40 mr-1">
          Trace
        </span>
        {steps.slice(-9).map((s, i) => (
          <span
            key={i}
            className="px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 text-slate-600 dark:text-white/60 whitespace-nowrap"
          >
            {s.label}
          </span>
        ))}
      </div>
      <div className="flex-1 overflow-auto px-3 py-2 font-mono text-[11px] leading-relaxed">
        {logs.length === 0 ? (
          <div className="text-slate-400 dark:text-white/25">
            Logs stream here as each node executes…
          </div>
        ) : (
          logs.map((l, i) => (
            <div key={i} className={lineColor(l.text)}>
              <span className="text-slate-300 dark:text-white/20 select-none">
                {String(i + 1).padStart(3, "0")}{" "}
              </span>
              {l.text}
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
