"""Scoring, Memory, and Complexity Tools for the ReTreVal Agent."""

import os
from typing import TYPE_CHECKING
from src.clients.llm_client import LLMClient
from src.utils.memory import ReflexionMemory

if TYPE_CHECKING:
    from src.agents.state import AgentState


class ScoringTool:
    """Scores nodes using local scoring with skepticism penalty from memory."""
    
    def __init__(self, llm: LLMClient, memory: ReflexionMemory, verbose: bool = True):
        self.llm = llm
        self.memory = memory
        self.verbose = verbose

    def invoke(self, state: "AgentState") -> "AgentState":
        nodes_map = state.get('tree', {}).get('nodes', {})
        curr_id = state.get('current_node_id')
        if not curr_id or curr_id not in nodes_map:
            return state
        
        res = self.llm.score_node(nodes_map[curr_id]['thought'])
        local_score = float(res.get('score', 0.5))
        nodes_map[curr_id]['local_score'] = local_score
        nodes_map[curr_id].setdefault('metadata', {})['score_rationale'] = res.get('rationale', '')
        
        # Apply skepticism penalty based on approach reliability from memory
        approach_type = state.get('approach_type', 'unknown')
        skepticism = self.memory.get_approach_skepticism(approach_type)
        
        # Reduce score based on approach history
        if skepticism > 0:
            local_score = local_score * (1.0 - skepticism)
            nodes_map[curr_id]['local_score'] = local_score
            if self.verbose:
                print(f"   ⚠️ Skepticism applied to '{approach_type}' approach: -{skepticism*100:.1f}%")
        
        # Combine with cross score
        local_w = float(os.getenv('LOCAL_SCORE_WEIGHT', '0.6'))
        cross_w = float(os.getenv('CROSS_SCORE_WEIGHT', '0.4'))
        ls = float(nodes_map[curr_id].get('local_score', 0.0))
        cs = float(nodes_map[curr_id].get('cross_score', 0.0))
        comb = local_w * ls + cross_w * cs
        nodes_map[curr_id]['combined_score'] = comb
        
        # Update state-level score tracking for current thought
        state['score'] = comb
        state['score_rationale'] = nodes_map[curr_id].get('metadata', {}).get('score_rationale', '')
        
        # Track best across all — prefer the refined node thought (which contains
        # the \boxed{} answer from the ReAct FINISH) over raw current_thought
        # (which may be the unrefined expansion description).
        if 'best_score' not in state or comb > state['best_score']:
            state['best_score'] = comb
            state['best_node_id'] = curr_id
            # The node's thought is updated by _react_refine_node with the
            # FINISH answer; use it as the authoritative best_thought.
            state['best_thought'] = nodes_map[curr_id].get('thought', '') or state.get('current_thought', '')
        
        if self.verbose:
            print("\n📊 SCORING PHASE (with skepticism adjustment)")
            print(f"Current node: {curr_id} | Score: {comb:.3f} | Best: {state.get('best_node_id','?')} {state.get('best_score',0.0):.3f}")
        
        return state


class MemoryTool:
    """Updates memory with insights and failures from recent nodes."""

    def __init__(self, llm: LLMClient, memory: ReflexionMemory, verbose: bool = True, cache_mgr=None):
        self.llm = llm
        self.memory = memory
        self.verbose = verbose
        self.cache_mgr = cache_mgr

    def invoke(self, state: "AgentState") -> "AgentState":
        # Build a minimal recent_nodes list using the current expansion's children
        nodes_map = state.get('tree', {}).get('nodes', {})
        ids = state.get('current_children_ids', []) or []
        recent = []
        for nid in ids:
            node = nodes_map.get(nid)
            if not node:
                continue
            recent.append({
                'id': nid,
                'thought': node.get('thought', ''),
                'combined_score': float(node.get('combined_score', 0.0)),
            })
        update = self.llm.update_memory(recent)
        for ins in update.get('insights', []):
            self.memory.add_insight(ins)
        for fail in update.get('failures', []):
            self.memory.add_failure(fail)
        if update.get('best_paths'):
            self.memory.update_best_paths(update['best_paths'])
        self.memory.increment_iteration()
        # Periodic consolidation every 5 iterations
        if self.memory.iteration_count % 5 == 0:
            self.memory.consolidate(llm_callable=self.llm._call_api)
        # Refresh cached memory summary (triggers prefix rebuild only if changed)
        if self.cache_mgr:
            state['memory_summary'] = self.cache_mgr.refresh_memory_summary()
        else:
            state['memory_summary'] = self.memory.get_summary()
        if self.verbose:
            print("\n🧠 MEMORY UPDATE")
            print(f"Insights: {len(update.get('insights', []))} | Failures: {len(update.get('failures', []))}")
        return state


class ComplexityTool:
    """Manages node iteration and selection strategy."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def invoke(self, state: "AgentState") -> "AgentState":
        # Rotate among current children, advancing to next child each iteration
        its = state.get('iterations', 0)
        thoughts = state.get('thoughts', [])
        ids = state.get('current_children_ids', []) or []
        if thoughts:
            idx = its % len(thoughts)
            state['current_thought'] = thoughts[idx]
            if idx < len(ids):
                state['current_node_id'] = ids[idx]
        state['iterations'] = its + 1
        if self.verbose:
            print("\n⚙️  ADAPTATION")
            print(f"Iteration: {state['iterations']}")
        return state
