"""
Shared prompts for MATH-500 evaluation to ensure consistency across all methods
"""

INITIAL_SOLVE_PROMPT = """Solve this math problem step by step with clear reasoning:

Problem: {question}

Provide a complete solution with all steps. At the end, put your final answer in \\boxed{{answer}} format.

Example format:
Step 1: ...
Step 2: ...
Therefore, the answer is \\boxed{{42}}"""

VERIFY_SOLUTION_PROMPT = """Verify this solution to the problem:

Problem: {question}
Solution: {solution}

Check if the solution is mathematically correct. List any potential errors or issues."""

REFINE_SOLUTION_PROMPT = """Improve this solution based on feedback:

Problem: {question}

Current Solution: {solution}

Feedback: {feedback}

Provide a corrected solution with clear steps. Put your final answer in \\boxed{{answer}} format."""

REFLECT_ON_SOLUTION_PROMPT = """Analyze this solution:

Problem: {question}
Solution: {solution}

Is this solution correct? What are potential errors or edge cases?"""

THOUGHT_PROMPT = """What is the key insight to solve this problem? Think step by step.

Problem: {question}

Identify the mathematical concepts and approach needed."""

OBSERVATION_PROMPT = """Solve this step by step:

Problem: {question}
Current Approach: {approach}

Provide a detailed solution with all steps. Put your final answer in \\boxed{{answer}} format."""

GENERATE_APPROACHES_PROMPT = """Given this math problem, generate {top_k} different solution approaches.

Problem: {question}

For each approach, explain the mathematical strategy briefly. Format as JSON:
[{{"approach_id": 1, "steps": "step1, step2, ..."}}, ...]

Provide ONLY the JSON array, no other text."""
