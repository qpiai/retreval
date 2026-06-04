"use client";

import { useEffect, useMemo } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  useReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { Tree, TreeNode } from "@/lib/types";

const NODE_W = 196;
const NODE_H = 84;
const X_GAP = 30;
const Y_GAP = 58;

function scoreColor(score: number): string {
  const s = Math.max(0, Math.min(1, score));
  if (s === 0) return "#64748b";
  if (s < 0.45) return "#f59e0b";
  if (s < 0.7) return "#eab308";
  return "#10b981";
}

type RFData = {
  label: string;
  thought: string;
  score: number;
  hasScore: boolean;
  isBest: boolean;
  isCurrent: boolean;
};

function ReTreNode({ data }: NodeProps<Node<RFData>>) {
  const color = scoreColor(data.score);
  return (
    <div
      className="relative rounded-xl px-3 py-2 text-left overflow-hidden border bg-white/90 dark:bg-bg-2/90 border-black/10 dark:border-white/10"
      style={{
        width: NODE_W,
        height: NODE_H,
        boxShadow: data.isBest
          ? "0 0 0 2px #ec4899, 0 8px 30px rgba(236,72,153,0.35)"
          : data.isCurrent
          ? "0 0 0 2px #8b5cf6, 0 8px 26px rgba(139,92,246,0.3)"
          : "0 4px 14px rgba(0,0,0,0.18)",
      }}
      title={data.thought}
    >
      <Handle type="target" position={Position.Top} className="!bg-edge-1 !border-0 !w-1.5 !h-1.5" />
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[11px] font-bold text-slate-900 dark:text-white/90">
          {data.label}
          {data.isBest && <span className="ml-1 text-accent">★</span>}
        </span>
        {data.hasScore && (
          <span
            className="font-mono text-[10px] font-semibold px-1.5 py-0.5 rounded"
            style={{ background: color, color: "#06121f" }}
          >
            {data.score.toFixed(2)}
          </span>
        )}
      </div>
      <div className="mt-1 text-[10px] leading-snug text-slate-500 dark:text-white/55 line-clamp-2">
        {data.thought || "…"}
      </div>
      {data.hasScore && (
        <div className="absolute bottom-0 left-0 h-1 w-full bg-black/5 dark:bg-white/5">
          <div
            className="h-full transition-all duration-500"
            style={{ width: `${data.score * 100}%`, background: color }}
          />
        </div>
      )}
      <Handle type="source" position={Position.Bottom} className="!bg-edge-1 !border-0 !w-1.5 !h-1.5" />
    </div>
  );
}

const nodeTypes = { retre: ReTreNode };

/** Tidy top-down layout: leaves laid left-to-right, parents centered. */
function layout(tree: Tree): { nodes: Node<RFData>[]; edges: Edge[] } {
  const get = (id: string): TreeNode | null =>
    id === "root" ? tree.root : tree.nodes[id] ?? null;

  if (!tree.root && Object.keys(tree.nodes).length === 0) {
    return { nodes: [], edges: [] };
  }

  const pos: Record<string, { x: number; depth: number }> = {};
  let leaf = 0;
  const seen = new Set<string>();

  const walk = (id: string, depth: number) => {
    if (seen.has(id)) return;
    seen.add(id);
    const n = get(id);
    if (!n) return;
    const kids = (n.children || []).filter((c) => get(c));
    if (kids.length === 0) {
      pos[id] = { x: leaf * (NODE_W + X_GAP), depth };
      leaf += 1;
    } else {
      kids.forEach((c) => walk(c, depth + 1));
      const xs = kids.map((c) => pos[c]?.x ?? 0);
      pos[id] = { x: (Math.min(...xs) + Math.max(...xs)) / 2, depth };
    }
  };
  walk("root", 0);

  Object.keys(tree.nodes).forEach((id) => {
    if (!pos[id]) {
      pos[id] = { x: leaf * (NODE_W + X_GAP), depth: tree.nodes[id].depth || 1 };
      leaf += 1;
    }
  });

  const nodes: Node<RFData>[] = [];
  const edges: Edge[] = [];
  const all: TreeNode[] = [
    ...(tree.root ? [tree.root] : []),
    ...Object.values(tree.nodes),
  ];

  for (const n of all) {
    const p = pos[n.id];
    if (!p) continue;
    const score = n.combined_score ?? 0;
    nodes.push({
      id: n.id,
      type: "retre",
      position: { x: p.x, y: p.depth * (NODE_H + Y_GAP) },
      data: {
        label: n.id === "root" ? "root" : n.id,
        thought: n.thought || "",
        score,
        hasScore: n.id !== "root" && score > 0,
        isBest: false,
        isCurrent: false,
      },
    });
    if (n.parent_id && get(n.parent_id)) {
      edges.push({
        id: `${n.parent_id}->${n.id}`,
        source: n.parent_id,
        target: n.id,
        style: { stroke: "#475569", strokeWidth: 1.5 },
      });
    }
  }
  return { nodes, edges };
}

function Flow({
  tree,
  bestNodeId,
  currentNodeId,
}: {
  tree: Tree;
  bestNodeId?: string | null;
  currentNodeId?: string | null;
}) {
  const rf = useReactFlow();
  const { nodes, edges } = useMemo(() => {
    const out = layout(tree);
    out.nodes = out.nodes.map((n) => ({
      ...n,
      data: {
        ...n.data,
        isBest: n.id === bestNodeId,
        isCurrent: n.id === currentNodeId,
      },
    }));
    // Highlight + animate edges leading into the current/best node.
    out.edges = out.edges.map((e) => {
      const hot = e.target === currentNodeId;
      const best = e.target === bestNodeId;
      return hot || best
        ? {
            ...e,
            animated: hot,
            style: {
              stroke: best ? "#ec4899" : "#8b5cf6",
              strokeWidth: 2.5,
            },
          }
        : e;
    });
    return out;
  }, [tree, bestNodeId, currentNodeId]);

  // Keep the whole tree framed as it grows / shifts.
  useEffect(() => {
    const id = window.setTimeout(
      () => rf.fitView({ padding: 0.22, duration: 400 }),
      60
    );
    return () => window.clearTimeout(id);
  }, [nodes.length, rf]);

  if (nodes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400 dark:text-white/25 text-sm px-6 text-center">
        The reasoning tree will grow here as the agent expands and scores
        candidate paths.
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.22 }}
      minZoom={0.15}
      maxZoom={1.6}
      proOptions={{ hideAttribution: true }}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#334155" />
      <Controls showInteractive={false} position="bottom-right" />
    </ReactFlow>
  );
}

export default function TreeView(props: {
  tree: Tree;
  bestNodeId?: string | null;
  currentNodeId?: string | null;
}) {
  return (
    <ReactFlowProvider>
      <Flow {...props} />
    </ReactFlowProvider>
  );
}
