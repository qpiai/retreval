"""
Tools Module
Centralized import and organization of all agent tools for clean separation of concerns

Structure:
  - feedback.py: ValidationTool, FailureAnalyzer, FeedbackLoopController, BacktrackingTool
  - search.py: WebSearchTool, SearchTool
  - planning.py: PlannerTool
  - refinement.py: RefinerTool, CritiqueTool
  - scoring.py: ScoringTool, MemoryTool, ComplexityTool
  - computation.py: PythonREPLTool, CalculatorTool, LinearProgrammingTool
  - synthesis.py: SynthesizerTool
  - registry.py: ToolSpec, ToolRegistry (for ReAct)
"""

# Import and re-export all tools
from .feedback import ValidationTool, FailureAnalyzer, FeedbackLoopController, BacktrackingTool
from .search import WebSearchTool, SearchTool
from .planning import PlannerTool
from .refinement import RefinerTool, CritiqueTool
from .scoring import ScoringTool, MemoryTool, ComplexityTool
from .computation import CalculatorTool, LinearProgrammingTool, PythonREPLTool
from .synthesis import SynthesizerTool
from .registry import ToolSpec, ToolRegistry

__all__ = [
    # Feedback & Validation
    'ValidationTool',
    'FailureAnalyzer',
    'FeedbackLoopController',
    'BacktrackingTool',
    # Search & Expansion
    'WebSearchTool',
    'SearchTool',
    # Planning
    'PlannerTool',
    # Refinement
    'RefinerTool',
    'CritiqueTool',
    # Scoring & Memory
    'ScoringTool',
    'MemoryTool',
    'ComplexityTool',
    # Computation
    'CalculatorTool',
    'LinearProgrammingTool',
    'PythonREPLTool',
    # Synthesis
    'SynthesizerTool',
    # ReAct Registry
    'ToolSpec',
    'ToolRegistry',
]
