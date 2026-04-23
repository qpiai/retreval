"""
Shared prompts for GSM8K evaluation to ensure consistency across all methods
"""

INITIAL_SOLVE_PROMPT = """Solve this math problem step by step with clear reasoning:

Problem: {question}

Provide a complete solution with all steps. At the end, clearly state the final answer in this format:
#### [final_answer_number]

Example format:
Step 1: ...
Step 2: ...
#### 42"""

VERIFY_SOLUTION_PROMPT = """Verify this solution to the problem:

Problem: {question}
Solution: {solution}

Check if the solution is correct and the answer makes sense. List any potential issues."""

REFINE_SOLUTION_PROMPT = """Improve this solution based on feedback:

Problem: {question}

Current Solution: {solution}

Feedback: {feedback}

Provide a corrected solution. At the end, clearly state the final answer as:
#### [final_answer_number]"""

REFLECT_ON_SOLUTION_PROMPT = """Analyze this solution:

Problem: {question}
Solution: {solution}

Is this solution correct? What are potential errors or edge cases?"""

THOUGHT_PROMPT = """What is the key insight to solve this problem? Think step by step.

Problem: {question}

Provide your reasoning in 2-3 sentences."""

ACTION_PROMPT = """Write Python code to solve this problem:

Problem: {question}

```python
# Solution code here
```"""

OBSERVATION_PROMPT = """Solve this step by step:

Problem: {question}
Current Approach: {approach}

Provide a detailed solution with all steps. At the end, state the final answer as:
#### [final_answer_number]"""

GENERATE_APPROACHES_PROMPT = """Given this math problem, generate {top_k} different solution approaches.

Problem: {question}

For each approach, explain the method briefly. Format as JSON:
[{{"approach_id": 1, "steps": "step1, step2, ..."}}, ...]

Provide ONLY the JSON array, no other text."""
