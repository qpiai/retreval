"""
CLI runner for the LangGraph Agent with configurable iterations and verbose output.
Supports both default LLMClient (Gemini/OpenAI) and Ollama client.
"""
import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.agent_langgraph import run_agent


def main():
    # Ensure .env is loaded from common locations
    try:
        load_dotenv()  # default search
        here = Path(__file__).resolve().parent
        load_dotenv(here / ".env")
        load_dotenv(here / "config.env")
    except Exception:
        pass
    
    parser = argparse.ArgumentParser(description="Run the LangGraph agent")
    parser.add_argument("--prompt", type=str, default="", help="Input prompt/problem")
    parser.add_argument("--iters", type=int, default=2, help="Number of refinement loops")
    parser.add_argument("--save", type=str, default="agent_result.json", help="Path to save result JSON")
    parser.add_argument("--quiet", action="store_true", help="Reduce console verbosity")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama client instead of default LLMClient")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name (e.g., qwen2.5:7b)")
    args = parser.parse_args()
    
    # Set up Ollama if requested
    if args.ollama:
        os.environ["USE_OLLAMA"] = "1"
        if args.model:
            os.environ["OLLAMA_MODEL"] = args.model
        print(f"🤖 Using Ollama client")
        if args.model:
            print(f"   Model: {args.model}")
        else:
            print(f"   Model: {os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')}")

    problem = args.prompt.strip()
    if not problem:
        print("Agent mode (LangGraph)")
        print("Enter your problem (single line; press Enter for default two-sum optimization):")
        user_in = input("Problem: ").strip()
        if user_in:
            problem = user_in
        else:
            problem = (
                "Optimize this code:\n\n"
                "def two_sum_bruteforce(nums, target):\n"
                "    for i in range(len(nums)):\n"
                "        for j in range(i + 1, len(nums)):\n"
                "            if nums[i] + nums[j] == target:\n"
                "                return [i, j]\n"
                "    return []\n"
            )

    result = run_agent(problem, iterations=max(1, int(args.iters)), verbose=(not args.quiet))
    # Print only the final answer to the terminal (no JSON dump)
    print(result.get('final_output', ''))
    if 'feasible' in result and result['feasible'] is False:
        print("\n[Note] The agent determined the problem is unsolvable under the given constraints.")
        if result.get('feasibility_reasons'):
            print("Reasons:", result['feasibility_reasons'])

    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.save}")

    # Save tree structure once, next to the result file for convenience
    try:
        tree_out = Path(args.save).with_name("tree_structure.json")
        tree_payload = result.get('tree', {})
        if tree_payload:
            with open(tree_out, "w", encoding="utf-8") as tf:
                json.dump(tree_payload, tf, indent=2)
            print(f"Saved tree: {tree_out}")
    except Exception as e:
        print(f"Warning: could not save tree_structure.json ({e})")


if __name__ == "__main__":
    main()
