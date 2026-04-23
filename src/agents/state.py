"""
Agent State Type Definitions

This file contains the AgentState TypedDict used by the LangGraph agent.
Separated to avoid circular import issues.
"""

from typing import List, Dict, Any, TypedDict


class AgentState(TypedDict, total=False):
    """State for the LangGraph agent tree search."""
    problem: str
    problem_id: str  # For tracking which problem
    expected_answer: str  # Ground truth answer for validation
    plan: str
    thoughts: List[str]
    current_thought: str
    current_node_id: str
    current_children_ids: List[str]
    critique_history: List[str]
    score: float
    score_rationale: str
    memory_summary: str
    recent_nodes: List[Dict[str, Any]]
    best_thought: str
    best_score: float
    best_node_id: str
    iterations: int
    final_output: str
    tree: Dict[str, Any]
    max_depth: int
    children_per_expansion: int
    complexity: int  # Problem complexity score (1-5)
    # Feedback loop fields
    is_correct: bool  # Was the final answer correct?
    predicted_answer: str  # The extracted answer from final_output
    error_analysis: str  # Root cause analysis of failure
    failure_type: str  # Category: arithmetic_error, logic_error, etc.
    approach_type: str  # Type of approach used: algebraic, arithmetic, etc.
    feedback_loops: int  # Number of retry loops executed
    should_retry: bool  # Should we continue iterating?
    # Backtracking fields
    failed_nodes: List[str]  # List of node IDs that have failed validation
    backtrack_history: List[Dict[str, Any]]  # History of backtracking events
    failure_contexts: List[Dict[str, Any]]  # Failure context from each failed node
    explored_root_children: List[str]  # Root children paths that have been explored
    backtrack_available: bool  # Is backtracking available as next option
    backtracking_iteration: int  # Counter for backtracking sub-iterations
    # Task decomposition fields
    subtasks: List[Dict[str, Any]]        # List of SubTask dicts
    subtask_results: Dict[str, str]       # sub_task_id -> result text
    is_decomposed: bool                    # Whether decomposition was applied
    is_subtask: bool                       # True = sub-agent context (prevents recursive decomposition)
    aggregated_result: str                 # Combined result from all sub-tasks
    # ReAct fields
    react_traces: List[Dict[str, Any]]     # Trace from ReAct loop per node
    tools_used: List[str]                  # Tools selected by LLM during ReAct
