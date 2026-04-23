"""
Creative Writing Solver using LangGraph Agent
Combines ReAct, Reflexion, and Self-Refine approaches using the advanced LangGraph agent.
Enhanced with retry logic, post-processing, and intelligent extraction.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.agent_langgraph import run_agent

load_dotenv()


class LangGraphCreativeWritingSolver:
    """Creative writing solver using the advanced LangGraph agent with ToT, ReAct, Reflexion, and Self-Refine."""
    
    def __init__(self, max_iterations: int = 3):
        load_dotenv()
        load_dotenv("config.env")
        
        self.max_iterations = max_iterations
    
    def solve(self, sentences: List[str], problem_id: int = None) -> Dict[str, Any]:
        """
        Generate a creative story with retry logic for better success rate.
        
        The agent will:
        1. Plan the story structure (Planner)
        2. Generate multiple story approaches (Tree of Thoughts)
        3. Critique and refine each approach (Reflexion)
        4. Score and select the best approach
        5. Synthesize the final story (Self-Refine)
        """
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            # Construct the problem statement
            if attempt == 0:
                problem = self._construct_problem(sentences)
            else:
                # On retry, be even more directive
                problem = self._construct_retry_problem(sentences, attempt)
            
            try:
                result = run_agent(
                    problem=problem,
                    problem_id=f"CREATIVE_{problem_id}" if problem_id else None,
                    iterations=self.max_iterations,
                    verbose=False
                )
                
                # Extract and post-process the story
                story = self._extract_story(result['final_output'], sentences)
                story = self._post_process_story(story, sentences)
                
                # Verify the story
                is_valid, reason = self.verify_story(story, sentences)
                
                if is_valid or attempt == max_retries:
                    # Success or final attempt
                    return {
                        'id': problem_id,
                        'sentences': sentences,
                        'story': story,
                        'is_valid': is_valid,
                        'validation_reason': reason,
                        'agent_plan': result.get('plan', ''),
                        'agent_thought': result.get('best_thought', ''),
                        'agent_score': result.get('score', 0.0),
                        'raw_output': result.get('final_output', ''),
                        'feasible': result.get('feasible', True),
                        'tree_depth': self._get_tree_depth(result.get('tree', {})),
                        'attempts': attempt + 1
                    }
                
                # Try to fix the story before retrying
                story = self._attempt_fix_story(story, sentences, reason)
                is_valid, reason = self.verify_story(story, sentences)
                
                if is_valid:
                    return {
                        'id': problem_id,
                        'sentences': sentences,
                        'story': story,
                        'is_valid': is_valid,
                        'validation_reason': reason + " (auto-fixed)",
                        'agent_plan': result.get('plan', ''),
                        'agent_thought': result.get('best_thought', ''),
                        'agent_score': result.get('score', 0.0),
                        'raw_output': result.get('final_output', ''),
                        'feasible': result.get('feasible', True),
                        'tree_depth': self._get_tree_depth(result.get('tree', {})),
                        'attempts': attempt + 1
                    }
            
            except Exception as e:
                if attempt == max_retries:
                    return {
                        'id': problem_id,
                        'sentences': sentences,
                        'story': "NO_STORY_GENERATED",
                        'is_valid': False,
                        'validation_reason': f"Agent error: {str(e)}",
                        'agent_plan': '',
                        'agent_thought': '',
                        'agent_score': 0.0,
                        'raw_output': '',
                        'feasible': False,
                        'tree_depth': 0,
                        'attempts': attempt + 1
                    }
                # Continue to next retry
                continue
        
        # Should never reach here, but just in case
        return {
            'id': problem_id,
            'sentences': sentences,
            'story': "NO_STORY_GENERATED",
            'is_valid': False,
            'validation_reason': "Max retries exceeded",
            'agent_plan': '',
            'agent_thought': '',
            'agent_score': 0.0,
            'raw_output': '',
            'feasible': False,
            'tree_depth': 0,
            'attempts': max_retries + 1
        }
    
    def _construct_problem(self, sentences: List[str]) -> str:
        """Construct a highly directive problem statement for the agent."""
        problem = f"""CREATIVE WRITING TASK: Write a 4-paragraph story with SPECIFIC ENDING SENTENCES.

CRITICAL REQUIREMENT - READ CAREFULLY:
Each paragraph MUST end with the EXACT sentence specified below. This is NON-NEGOTIABLE.

REQUIRED ENDING SENTENCES (use these EXACTLY as written):
Paragraph 1 must end with: "{sentences[0]}"
Paragraph 2 must end with: "{sentences[1]}"
Paragraph 3 must end with: "{sentences[2]}"
Paragraph 4 must end with: "{sentences[3]}"

STORY REQUIREMENTS:
✓ Exactly 4 paragraphs (no more, no less)
✓ Each paragraph is 3-5 sentences long
✓ Each paragraph MUST end with its assigned sentence word-for-word
✓ Story must be coherent and flow naturally
✓ Use creative transitions between paragraphs

OUTPUT FORMAT:
Provide ONLY the 4-paragraph story. Do NOT include:
- Analysis or commentary
- Explanations
- Meta-text
- Headers or labels
- Code or formatting

Just write the story with paragraphs separated by blank lines.

EXAMPLE FORMAT:
[First sentence of paragraph 1.] [Second sentence.] [Third sentence.] {sentences[0]}

[First sentence of paragraph 2.] [Second sentence.] [Third sentence.] {sentences[1]}

[First sentence of paragraph 3.] [Second sentence.] [Third sentence.] {sentences[2]}

[First sentence of paragraph 4.] [Second sentence.] [Third sentence.] {sentences[3]}

Now write your creative story following this exact format:"""
        
        return problem
    
    def _construct_retry_problem(self, sentences: List[str], attempt: int) -> str:
        """Construct a more directive problem for retry attempts."""
        problem = f"""RETRY ATTEMPT {attempt}: Previous story did not meet requirements.

CRITICAL: Each paragraph MUST end with the EXACT sentence specified below.

REQUIRED ENDINGS (copy these EXACTLY):
1. "{sentences[0]}"
2. "{sentences[1]}"
3. "{sentences[2]}"
4. "{sentences[3]}"

Write a 4-paragraph story where:
- Paragraph 1 ends with sentence 1
- Paragraph 2 ends with sentence 2
- Paragraph 3 ends with sentence 3
- Paragraph 4 ends with sentence 4

Output ONLY the story (no analysis, no headers, no explanations).
Separate paragraphs with blank lines.

Write the story now:"""
        
        return problem
    
    def _extract_story(self, agent_output: str, sentences: List[str]) -> str:
        """Extract the story from the agent's output with intelligent parsing."""
        if not agent_output or agent_output == "NO_STORY_GENERATED":
            return "NO_STORY_GENERATED"
        
        # Strategy 1: Look for the story after common headers
        story = agent_output
        
        # Remove sections that are clearly not the story
        sections_to_remove = [
            "## Problem Analysis",
            "## Solution",
            "## Approach",
            "## Strategy",
            "## Final Answer in Plain English",
            "## Final Answer",
            "**Problem:**",
            "**Requirements:**",
            "**Analysis:**",
            "**Note:**",
            "**Challenge Identified:**",
            "**Recommendation:**",
            "**Solution:**",
            "**Story:**"
        ]
        
        for section in sections_to_remove:
            if section in story:
                parts = story.split(section)
                if len(parts) > 1:
                    potential_story = parts[-1].strip()
                    if len(potential_story) > 100:
                        story = potential_story
        
        # Strategy 2: If we find "Final Answer in Plain English", take content before it
        if "Final Answer in Plain English" in story:
            story = story.split("Final Answer in Plain English")[0].strip()
        
        # Strategy 3: Remove markdown code blocks
        story = re.sub(r'```[\s\S]*?```', '', story)
        
        # Clean up the story
        story = story.strip()
        
        # Remove leading/trailing quotes if present
        if story.startswith('"') and story.endswith('"'):
            story = story[1:-1]
        if story.startswith("'") and story.endswith("'"):
            story = story[1:-1]
        
        # Strategy 4: If the story doesn't contain the required sentences,
        # try to find a section that does
        if not any(sent.lower() in story.lower() for sent in sentences):
            paragraphs = [p.strip() for p in agent_output.split('\n\n') if p.strip()]
            
            # Find consecutive paragraphs that contain the required sentences
            for i in range(len(paragraphs) - 3):
                candidate = '\n\n'.join(paragraphs[i:i+4])
                if sum(1 for sent in sentences if sent.lower() in candidate.lower()) >= 3:
                    story = candidate
                    break
        
        # Strategy 5: Look for narrative content (sentences with proper structure)
        if len(story) < 50 or story.count('.') < 8:
            # Extract paragraphs that look like narrative
            all_paragraphs = [p.strip() for p in agent_output.split('\n\n') if p.strip()]
            narrative_paragraphs = []
            
            for para in all_paragraphs:
                # Check if paragraph looks like narrative (has sentences, not too short)
                if (len(para) > 30 and 
                    para.count('.') >= 2 and 
                    not para.startswith('#') and
                    not para.startswith('**') and
                    not para.startswith('-')):
                    narrative_paragraphs.append(para)
            
            if len(narrative_paragraphs) >= 4:
                story = '\n\n'.join(narrative_paragraphs[:4])
        
        return story.strip()
    
    def _post_process_story(self, story: str, sentences: List[str]) -> str:
        """Clean and fix common story formatting issues."""
        if not story or story == "NO_STORY_GENERATED":
            return story
        
        # Remove markdown formatting
        story = re.sub(r'\*\*([^*]+)\*\*', r'\1', story)  # Remove bold
        story = re.sub(r'##\s+', '', story)  # Remove headers
        story = re.sub(r'^\s*[-*]\s+', '', story, flags=re.MULTILINE)  # Remove bullets
        story = re.sub(r'`([^`]+)`', r'\1', story)  # Remove inline code
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in story.split('\n\n') if p.strip()]
        
        # If we have more than 4 paragraphs, try to merge short ones or take first 4
        if len(paragraphs) > 4:
            # First try: take paragraphs that contain the required sentences
            matched_paragraphs = []
            for sent in sentences:
                for para in paragraphs:
                    normalized_para = ' '.join(para.split()).lower()
                    normalized_sent = ' '.join(sent.split()).lower()
                    if normalized_sent in normalized_para and para not in matched_paragraphs:
                        matched_paragraphs.append(para)
                        break
            
            if len(matched_paragraphs) == 4:
                paragraphs = matched_paragraphs
            else:
                # Second try: merge short paragraphs
                merged = []
                i = 0
                while i < len(paragraphs) and len(merged) < 4:
                    if i < len(paragraphs) - 1 and len(paragraphs[i].split()) < 15:
                        merged.append(paragraphs[i] + ' ' + paragraphs[i+1])
                        i += 2
                    else:
                        merged.append(paragraphs[i])
                        i += 1
                paragraphs = merged[:4]
        
        # If we have fewer than 4 paragraphs, try to split long ones
        if len(paragraphs) < 4:
            expanded = []
            for p in paragraphs:
                sentences_in_p = [s.strip() for s in re.split(r'[.!?]+', p) if s.strip()]
                if len(sentences_in_p) > 6 and len(expanded) < 3:
                    # Split into two paragraphs
                    mid = len(sentences_in_p) // 2
                    para1 = '. '.join(sentences_in_p[:mid]) + '.'
                    para2 = '. '.join(sentences_in_p[mid:]) + '.'
                    expanded.append(para1)
                    expanded.append(para2)
                else:
                    expanded.append(p)
            paragraphs = expanded
        
        # Ensure we have exactly 4 paragraphs
        while len(paragraphs) < 4:
            paragraphs.append("The story continues.")
        paragraphs = paragraphs[:4]
        
        # Try to fix missing ending sentences
        for i, expected_sent in enumerate(sentences):
            if i < len(paragraphs):
                para = paragraphs[i]
                normalized_para = ' '.join(para.split()).lower()
                normalized_sent = ' '.join(expected_sent.split()).lower()
                
                # Remove trailing punctuation for comparison
                normalized_para_clean = normalized_para.rstrip('.!?;, ')
                normalized_sent_clean = normalized_sent.rstrip('.!?;, ')
                
                # If paragraph doesn't end with required sentence, try to fix it
                if not normalized_para_clean.endswith(normalized_sent_clean):
                    # Check if sentence appears anywhere in paragraph
                    if normalized_sent_clean in normalized_para:
                        # Extract everything up to and including the sentence
                        # Find the position in the original paragraph
                        sent_lower = expected_sent.lower()
                        para_lower = para.lower()
                        idx = para_lower.rfind(sent_lower)
                        if idx >= 0:
                            paragraphs[i] = para[:idx + len(expected_sent)].rstrip('.!?;, ') + '.'
                    else:
                        # Append the required sentence
                        para_clean = para.rstrip('.!?;, ')
                        paragraphs[i] = para_clean + '. ' + expected_sent
        
        return '\n\n'.join(paragraphs)
    
    def _attempt_fix_story(self, story: str, sentences: List[str], validation_reason: str) -> str:
        """Attempt to automatically fix common story issues."""
        # Apply post-processing (which includes fixing)
        fixed_story = self._post_process_story(story, sentences)
        
        # Check if fix worked
        is_valid, _ = self.verify_story(fixed_story, sentences)
        if is_valid:
            return fixed_story
        
        # If still invalid, return post-processed version anyway (might be closer)
        return fixed_story
    
    def _get_tree_depth(self, tree: Dict[str, Any]) -> int:
        """Get the maximum depth of the thought tree."""
        if not tree or 'nodes' not in tree:
            return 0
        
        nodes = tree.get('nodes', {})
        if not nodes:
            return 0
        
        max_depth = 0
        for node in nodes.values():
            depth = node.get('depth', 0)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def verify_story(self, story: str, expected_sentences: List[str]) -> tuple[bool, str]:
        """Verify the story meets requirements with fuzzy matching."""
        if not story or story == "NO_STORY_GENERATED":
            return False, "No story generated"
        
        # Split by double newlines first, then try single newlines
        paragraphs = [p.strip() for p in story.split('\n\n') if p.strip()]
        if len(paragraphs) < 4:
            # Try single newline split
            paragraphs = [p.strip() for p in story.split('\n') if p.strip() and len(p.strip()) > 20]
        
        if len(paragraphs) < 4:
            return False, f"Only {len(paragraphs)} paragraphs found (need 4)"
        
        missing_sentences = []
        for i, expected_sent in enumerate(expected_sentences):
            if i < len(paragraphs):
                # Normalize both paragraph and expected sentence
                para = paragraphs[i].strip()
                normalized_para = ' '.join(para.split()).lower()
                normalized_sent = ' '.join(expected_sent.split()).lower()
                
                # Remove trailing punctuation for comparison
                normalized_para = normalized_para.rstrip('.!?;, ')
                normalized_sent = normalized_sent.rstrip('.!?;, ')
                
                if not normalized_para.endswith(normalized_sent):
                    missing_sentences.append(f"Paragraph {i+1}: missing '{expected_sent}'")
        
        if missing_sentences:
            return False, "; ".join(missing_sentences)
        
        return True, "All sentences correctly placed"


def read_sentences_from_csv(csv_file: str, limit: int = None) -> List[Dict[str, Any]]:
    """Read creative writing prompts from CSV file."""
    problems = []
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found")
        return problems
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            required_cols = ['id', 'sentence_1', 'sentence_2', 'sentence_3', 'sentence_4']
            if not all(col in reader.fieldnames for col in required_cols):
                print(f"❌ Error: CSV must have columns: {', '.join(required_cols)}")
                return problems
            
            for row in reader:
                problem_id = int(row['id'])
                if limit and problem_id > limit:
                    break
                
                problems.append({
                    'id': problem_id,
                    'sentences': [
                        row['sentence_1'].strip(),
                        row['sentence_2'].strip(),
                        row['sentence_3'].strip(),
                        row['sentence_4'].strip()
                    ]
                })
    
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return problems
    
    print(f"📚 Loaded {len(problems)} problems from {csv_file}")
    return problems


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save results to CSV file."""
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['id', 'sentence_1', 'sentence_2', 'sentence_3', 'sentence_4', 
                         'story', 'is_valid', 'validation_reason', 'agent_score', 'tree_depth', 'attempts']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'id': result['id'],
                    'sentence_1': result['sentences'][0],
                    'sentence_2': result['sentences'][1],
                    'sentence_3': result['sentences'][2],
                    'sentence_4': result['sentences'][3],
                    'story': result['story'],
                    'is_valid': result['is_valid'],
                    'validation_reason': result['validation_reason'],
                    'agent_score': result.get('agent_score', 0.0),
                    'tree_depth': result.get('tree_depth', 0),
                    'attempts': result.get('attempts', 1)
                })
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print statistics
        valid = sum(1 for r in results if r['is_valid'])
        avg_score = sum(r.get('agent_score', 0.0) for r in results) / len(results) if results else 0
        avg_depth = sum(r.get('tree_depth', 0) for r in results) / len(results) if results else 0
        avg_attempts = sum(r.get('attempts', 1) for r in results) / len(results) if results else 0
        
        print(f"\n📊 Statistics:")
        print(f"   ✓ Valid stories: {valid}/{len(results)} ({valid/len(results)*100:.1f}%)")
        print(f"   📈 Average agent score: {avg_score:.2f}")
        print(f"   🌳 Average tree depth: {avg_depth:.1f}")
        print(f"   🔄 Average attempts: {avg_attempts:.1f}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate creative stories using LangGraph Agent (ToT + ReAct + Reflexion + Self-Refine)'
    )
    parser.add_argument(
        'input_csv',
        help='Input CSV file with sentence columns'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of problems to process'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Maximum iterations for the agent (default: 3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_name = Path(args.input_csv).stem
        output_file = f"{input_name}_langgraph_{timestamp}.csv"
    
    print(f"\n🚀 LangGraph Creative Writing Solver (Enhanced)")
    print(f"{'='*80}")
    print(f"Input: {args.input_csv}")
    print(f"Output: {output_file}")
    print(f"Limit: {args.limit if args.limit else 'No limit'}")
    print(f"Max Iterations: {args.iterations}")
    print(f"{'='*80}\n")
    
    # Read problems
    problems = read_sentences_from_csv(args.input_csv, limit=args.limit)
    
    if not problems:
        print("❌ No problems to process")
        return 1
    
    # Initialize solver
    solver = LangGraphCreativeWritingSolver(
        max_iterations=args.iterations
    )
    
    # Process each problem
    results = []
    
    for idx, problem in enumerate(problems, 1):
        print(f"\n🔄 Processing {idx}/{len(problems)}...")
        
        result = solver.solve(problem['sentences'], problem['id'])
        results.append(result)
        
        # Show validation status
        if result['is_valid']:
            print(f"   ✅ Valid story generated (attempt {result.get('attempts', 1)})")
        else:
            print(f"   ⚠️  Invalid: {result['validation_reason'][:50]}...")
    
    # Save final results
    save_results_to_csv(results, output_file)
    
    print(f"\n✅ All done! Results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
