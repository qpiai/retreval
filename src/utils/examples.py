"""
Custom Example Runner: Supports both Tree-of-Thoughts (ToT) and Agentic (LangGraph) modes.
Select mode via CLI flag --agent or environment variable USE_AGENT=1.
"""
import os
import json
import argparse
try:
    # Agent mode (optional)
    from agent_langgraph import run_agent as run_agent_fn
except Exception:
    run_agent_fn = None


def run_problem(problem: str, max_iterations: int, use_agent: bool):
    """Run either ToT ReasoningEngine or Agentic LangGraph based on flag."""
    if use_agent:
        if run_agent_fn is None:
            raise RuntimeError("Agent mode requested but agent_langgraph is unavailable.")
        print("\n" + "🤖 "*20)
        print("AGENTIC MODE (LangGraph)")
        print("🤖 "*20 + "\n")
        result = run_agent_fn(problem, iterations=max_iterations, verbose=True)
        return result
    else:
        try:
            from main import ReasoningEngine  # Lazy import so agent-only mode doesn't require main.py
        except Exception:
            raise RuntimeError("ToT mode requested but main.py/ReasoningEngine is unavailable. Use --agent or restore main.py.")
        engine = ReasoningEngine()
        result = engine.reason(problem, max_iterations=max_iterations, verbose=True)
        return result


def example_1_route_planning(use_agent: bool, iters: int):
    """Example: Complex route planning with constraints."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Route Planning Problem")
    print("="*70 + "\n")
    
    problem = """
    You need to travel from City A to Island X. There are multiple routes:
    1. Direct ferry (expensive, weather-dependent)
    2. Drive to Port B, then ferry (cheaper, longer)
    3. Drive to Port C via Highway (scenic but may have traffic)
    4. Mixed transport with transfers
    
    Recent floods have affected some roads. What is the best route considering 
    safety, cost, and reliability?
    """
    
    result = run_problem(problem, max_iterations=iters or 8, use_agent=use_agent)
    
    return result


def example_2_business_strategy(use_agent: bool, iters: int):
    """Example: Business strategy decision."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Business Strategy Problem")
    print("="*70 + "\n")
    
    problem = """
    A tech startup has $500K in funding and 6 months runway. They need to decide:
    
    Option A: Focus on B2B enterprise sales (high value, slow sales cycle)
    Option B: Launch B2C product (fast growth, uncertain monetization)
    Option C: Hybrid approach (split resources)
    Option D: Pivot to consulting to extend runway
    
    Market conditions: Competition is high, but there's a niche opportunity in 
    healthcare AI. Team has strong technical skills but limited sales experience.
    
    What strategy maximizes chances of success?
    """
    
    result = run_problem(problem, max_iterations=iters or 10, use_agent=use_agent)
    
    return result


def example_3_technical_architecture(use_agent: bool, iters: int):
    """Example: Technical architecture decision."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Technical Architecture Problem")
    print("="*70 + "\n")
    
    problem = """
    Design a scalable backend for a real-time chat application expecting:
    - 100K concurrent users in 6 months
    - Message delivery latency < 100ms
    - 99.9% uptime requirement
    - Limited budget ($2K/month)
    
    Consider:
    - Database choice (SQL vs NoSQL vs hybrid)
    - Message queue system
    - Caching strategy
    - Deployment (cloud provider, regions)
    - Monitoring and scaling approach
    
    What architecture would you recommend?
    """
    
    result = run_problem(problem, max_iterations=iters or 10, use_agent=use_agent)
    
    return result


def example_4_scientific_reasoning(use_agent: bool, iters: int):
    """Example: Scientific hypothesis generation."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Scientific Reasoning Problem")
    print("="*70 + "\n")
    
    problem = """
    Observation: A research team found that patients taking Drug X for condition Y
    showed unexpected improvement in condition Z (unrelated to Y).
    
    Known facts:
    - Drug X targets protein P1
    - Condition Z involves inflammation and protein P2
    - P1 and P2 share 40% structural similarity
    - 60% of patients showed improvement, 40% showed no change
    
    Generate hypotheses to explain this observation and suggest experiments to test them.
    What is the most likely mechanism and how would you validate it?
    """
    
    result = run_problem(problem, max_iterations=iters or 12, use_agent=use_agent)
    
    return result


def save_results(results, filename="all_examples_results.json"):
    """Save all results to a file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 All results saved to {filename}")


def main():
    """Run selected examples in ToT or Agentic mode."""
    parser = argparse.ArgumentParser(description="Run examples in ToT or Agent mode")
    parser.add_argument("--agent", action="store_true", help="Use agentic (LangGraph) mode")
    parser.add_argument("--iters", type=int, default=0, help="Max iterations per example")
    parser.add_argument("--all", action="store_true", help="Run all examples (otherwise runs Example 1 only)")
    parser.add_argument("--save", type=str, default="all_examples_results.json", help="Output JSON filename")
    args = parser.parse_args()

    use_agent = args.agent or os.getenv("USE_AGENT", "0") == "1"

    print("\n" + "🌳 "*30)
    print("Examples Runner: ToT + Agentic")
    print("Provider via .env (LLM_PROVIDER)")
    print("🌳 "*30 + "\n")

    results = {}

    # Default: only Example 1, unless --all is passed
    results['route_planning'] = example_1_route_planning(use_agent, args.iters)

    if args.all:
        # Uncommented when --all provided
        results['business_strategy'] = example_2_business_strategy(use_agent, args.iters)
        results['technical_architecture'] = example_3_technical_architecture(use_agent, args.iters)
        results['scientific_reasoning'] = example_4_scientific_reasoning(use_agent, args.iters)

    save_results(results, filename=args.save)
    
    print("\n" + "✅ "*30)
    print("Examples completed!")
    print("✅ "*30 + "\n")


if __name__ == "__main__":
    main()
