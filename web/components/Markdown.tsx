"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";

/**
 * Normalize the LaTeX the agent emits so KaTeX (via remark-math) can render it:
 *   \[ … \]  → $$ … $$        (display math)
 *   \( … \)  → $ … $          (inline math)
 *   bare \boxed{…}            → wrapped in $…$ so it renders instead of showing raw
 * Properly-delimited $…$ / $$…$$ are left untouched.
 */
function mathify(src: string): string {
  if (!src) return src;
  let t = src;
  // \[ … \] → $$ … $$   ·   \( … \) → $ … $
  t = t.replace(/\\\[([\s\S]+?)\\\]/g, (_m, e) => `$$${e}$$`);
  t = t.replace(/\\\(([\s\S]+?)\\\)/g, (_m, e) => `$${e}$`);
  // \boxed{…} not already delimited → wrap so KaTeX renders it
  t = t.replace(/(?<!\$)\\boxed\{([^{}]+)\}(?!\$)/g, (_m, e) => `$\\boxed{${e}}$`);
  // If the model emitted bare LaTeX macros without $ delimiters anywhere, wrap
  // the common ones so they still render (e.g. "P = \frac{2}{15}" in prose).
  if (!/\$/.test(t)) {
    t = t.replace(/\\frac\s*\{[^{}]*\}\s*\{[^{}]*\}/g, (m) => `$${m}$`);
    t = t.replace(
      /\\(?:sqrt|text|overline|vec|hat|bar)\s*\{[^{}]*\}/g,
      (m) => `$${m}$`
    );
  }
  return t;
}

export default function Markdown({
  children,
  className = "prose-rt",
  inline = false,
}: {
  children: string;
  className?: string;
  inline?: boolean;
}) {
  const md = (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex, rehypeHighlight]}
      components={
        inline
          ? { p: ({ children }) => <>{children}</> }
          : undefined
      }
    >
      {mathify(children || "")}
    </ReactMarkdown>
  );
  return inline ? <span className={className}>{md}</span> : <div className={className}>{md}</div>;
}
