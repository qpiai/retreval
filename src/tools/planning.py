"""
Planning Tool
Handles problem analysis and tree initialization
"""

import re
from typing import Dict, Any

from src.agents.state import AgentState
from src.clients.llm_client import LLMClient
from src.utils.memory import Memory


class PlannerTool:
    """Plan problem solving strategy and assess complexity for adaptive tree depth."""

    def __init__(self, llm: LLMClient, memory: Memory, verbose: bool = True, cache_mgr=None):
        self.llm = llm
        self.memory = memory
        self.verbose = verbose
        self.cache_mgr = cache_mgr

    def _assess_complexity(self, problem: str) -> int:
        """
        Assess problem complexity using LLM to determine adaptive tree depth.
        
        Returns:
            complexity_score: 1 (simple) to 5 (very complex)
        """
        prompt = f"""Analyze the complexity of this problem and rate it from 1 (very simple) to 5 (very complex).

Problem: {problem}

Consider:
- Number of constraints or variables
- Mathematical difficulty
- Number of steps required
- Ambiguity or vagueness

Respond with ONLY a single digit (1-5) and a brief reason.
Format: "Complexity: X - [reason]"
"""
        response = self.llm._call_api(prompt) or "Complexity: 3 - medium"
        
        # Extract complexity score
        match = re.search(r'Complexity:\s*(\d)', response)
        if match:
            score = int(match.group(1))
            return min(max(score, 1), 5)  # Clamp 1-5
        return 3  # Default to medium

    def invoke(self, state: AgentState) -> AgentState:
        # Use relevant memory instead of full dump
        if self.cache_mgr:
            memo = self.memory.get_relevant(state['problem'], max_entries=3) or "No memory insights yet."
        else:
            memo = self.memory.get_summary()
        
        # Assess complexity to determine tree depth adaptively
        complexity = self._assess_complexity(state['problem'])
        
        # Map complexity to tree depth and branching
        complexity_config = {
            1: {'max_depth': 1, 'children_per_expansion': 2},
            2: {'max_depth': 2, 'children_per_expansion': 2},
            3: {'max_depth': 3, 'children_per_expansion': 3},
            4: {'max_depth': 4, 'children_per_expansion': 3},
            5: {'max_depth': 5, 'children_per_expansion': 4},
        }
        
        config = complexity_config.get(complexity, {'max_depth': 2, 'children_per_expansion': 3})
        
        if self.verbose:
            print(f"\n📊 Problem Complexity Assessment: {complexity}/5")
            print(f"🌳 Adaptive Tree Config: max_depth={config['max_depth']}, children_per_expansion={config['children_per_expansion']}")
        
        prompt = f"""You are an expert mathematical problem solver creating a strategic plan.

Problem:
{state['problem']}

Memory insights:
{memo}

For mathematical problems:
1. Identify the target value and available numbers/operations
2. Consider different groupings using parentheses
3. Try various operator combinations systematically
4. Verify arithmetic at each step
5. Check all constraints are satisfied

Return 3-5 concrete, actionable steps focused on solving THIS specific problem."""
        plan = self.llm._call_api(prompt) or ""
        state['plan'] = plan.strip()
        state['approach_type'] = 'general'  # Set default so skepticism system works
        state['memory_summary'] = memo

        # Update KV cache manager with the plan for prefix reuse
        if self.cache_mgr:
            self.cache_mgr.update_plan(state['plan'])

        # Store complexity score
        state['complexity'] = complexity

        # Initialize tree structure
        root_node = {
            "id": "root",
            "parent_id": None,
            "depth": 0,
            "thought": f"IMPROVED:\n{state['plan']}",
            "state": {"type": "problem"},
            "refinement_count": 0,
            "critique_history": [],
            "local_score": 1.0,
            "cross_score": 0.0,
            "combined_score": 0.6,
            "children": [],
            "provenance": {"method": "initialization"},
            "metadata": {"score_rationale": "Initial root node", "complexity": complexity},
        }
        state['tree'] = {"root": root_node, "nodes": {}}

        # Set adaptive depth/branching config based on complexity, capped by env var
        import os
        max_depth_cap = int(os.getenv('MAX_DEPTH', '5'))
        state['max_depth'] = min(config['max_depth'], max_depth_cap)
        state['children_per_expansion'] = config['children_per_expansion']
        
        return state
