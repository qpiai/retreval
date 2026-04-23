"""Refinement and Critique Tools for the ReTreVal Agent."""

from typing import TYPE_CHECKING
from src.clients.llm_client import LLMClient
from src.utils.memory import ReflexionMemory

if TYPE_CHECKING:
    from src.agents.state import AgentState


class RefinerTool:
    """Self-refines a thought based on critique history and memory insights."""

    def __init__(self, llm: LLMClient, memory: ReflexionMemory, verbose: bool = True):
        self.llm = llm
        self.memory = memory
        self.verbose = verbose

    # ---------- ReAct call() wrapper ----------
    def call(self, critique: str, state: dict = None) -> str:
        """Refine the current reasoning by addressing *critique*.

        When called from ReAct, *state* is bound via closure and provides
        the current thought and critique history.
        """
        try:
            current_thought = (state or {}).get('current_thought', '')
            critique_history = list((state or {}).get('critique_history', []))
            critique_history.append(critique)

            refined = self.llm.self_refine(
                thought=current_thought,
                critique_history=critique_history,
                memory_summary=self.memory.get_summary(),
            )
            improved = refined.get('improved_thought', current_thought)

            # Side-effect: update state if provided
            if state is not None:
                state['current_thought'] = improved
                state.setdefault('critique_history', []).append(refined.get('critique', critique))

            return f"Refined reasoning:\n{improved[:600]}"
        except Exception as exc:
            return f"Refine error: {exc}"

    def invoke(self, state: "AgentState") -> "AgentState":
        crt = state.get('critique_history', []) or []
        refined = self.llm.self_refine(
            thought=state['current_thought'],
            critique_history=crt,
            memory_summary=self.memory.get_summary(),
        )
        state['current_thought'] = refined['improved_thought']
        state.setdefault('critique_history', []).append(refined['critique'])
        # Update node in tree
        nid = state.get('current_node_id')
        nodes_map = state.get('tree', {}).get('nodes', {})
        if nid in nodes_map:
            node = nodes_map[nid]
            node['thought'] = refined['improved_thought']
            node.setdefault('critique_history', []).append(refined['critique'])
            node['refinement_count'] = int(node.get('refinement_count', 0)) + 1
        if self.verbose:
            print("\n✨ REFINEMENT PHASE")
            print(f"IMPROVED: {refined['improved_thought'][:200]}")
        return state


class CritiqueTool:
    """Cross-critique tool that compares all pairs of nodes and assigns scores."""
    
    def __init__(self, llm: LLMClient, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose

    def invoke(self, state: "AgentState") -> "AgentState":
        thoughts = state.get('thoughts', [])
        ids = state.get('current_children_ids', [])
        nodes_map = state.get('tree', {}).get('nodes', {})
        
        if len(thoughts) < 2:
            return state
        
        # Initialize score accumulators for each node
        node_scores = {i: [] for i in range(len(thoughts))}
        
        # Perform pairwise comparisons for ALL node combinations
        for i in range(len(thoughts)):
            for j in range(i + 1, len(thoughts)):
                # Generate labels (A, B, C, D, etc.)
                label_i = chr(65 + i)
                label_j = chr(65 + j)
                
                # Compare node i with node j
                scores = self.llm.cross_critique(label_i, thoughts[i], label_j, thoughts[j])
                
                # Accumulate scores for both nodes
                node_scores[i].append(float(scores.get(label_i, 0.0)))
                node_scores[j].append(float(scores.get(label_j, 0.0)))
        
        # Calculate average cross_score for each node
        cross_scores = {}
        best_idx = 0
        best_score = 0.0
        
        for i in range(len(thoughts)):
            if node_scores[i]:
                avg_score = sum(node_scores[i]) / len(node_scores[i])
            else:
                avg_score = 0.0
            
            cross_scores[i] = avg_score
            
            # Find best node by average cross_score
            if avg_score > best_score:
                best_score = avg_score
                best_idx = i
        
        # Store cross scores on ALL children (not just first two)
        for i, node_id in enumerate(ids):
            if node_id in nodes_map:
                nodes_map[node_id]['cross_score'] = cross_scores.get(i, 0.0)
        
        # Select the best node
        state['current_thought'] = thoughts[best_idx]
        if best_idx < len(ids):
            state['current_node_id'] = ids[best_idx]
        
        if self.verbose:
            print("\n🔍 CRITIQUE PHASE (ALL-NODES COMPARISON)")
            labels = [chr(65 + i) for i in range(len(thoughts))]
            score_info = []
            for i, label in enumerate(labels):
                score_val = cross_scores.get(i, 0.0)
                comparisons = len(node_scores.get(i, []))
                score_info.append(f"  {label}: {score_val:.3f} (from {comparisons} comparisons)")
            print("\n".join(score_info))
            best_label = chr(65 + best_idx)
            print(f"\n✓ Selected: {best_label} (score: {cross_scores.get(best_idx, 0.0):.3f})")
            print(f"  {state['current_thought'][:200]}")
        
        return state
