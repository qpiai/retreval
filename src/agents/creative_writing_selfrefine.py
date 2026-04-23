"""
Creative Writing Solver using Self-Refine
Self-Refine iteratively generates and improves stories based on feedback.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Use OpenAI as primary, fallback to Ollama if needed
provider = os.getenv("LLM_PROVIDER", "openai").lower()

if provider == "openai":
    from src.clients.openai_client import OpenAIClient
    LLMClient = OpenAIClient
elif provider == "ollama":
    from src.clients.ollama_client import OllamaClient
    LLMClient = OllamaClient
else:
    from src.clients.openai_client import OpenAIClient
    LLMClient = OpenAIClient

load_dotenv()


class SelfRefineCreativeWritingSolver:
    """Self-Refine solver that generates and iteratively improves stories."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        load_dotenv()
        load_dotenv("config.env")
        
        self.llm = LLMClient()
        
        self.max_refinements = int(os.getenv('MAX_LOOPS', '3'))
    
    def solve(self, sentences: List[str], problem_id: int = None) -> Dict[str, Any]:
        """
        Generate a creative story using Self-Refine approach.
        
        Self-Refine Pattern:
        1. Generate an initial story
        2. Critique the story (check constraints, flow, creativity)
        3. Refine the story based on critique
        4. Repeat until satisfactory or max refinements
        """
        if self.verbose:
            id_str = f" [{problem_id}]" if problem_id else ""
            print(f"\n{'='*80}")
            print(f"🔄 Self-Refine Creative Writing Solver{id_str}")
            print(f"{'='*80}")
            for i, sent in enumerate(sentences, 1):
                print(f"  {i}. {sent}")
        
        # Step 1: Generate initial story
        if self.verbose:
            print(f"\n--- Initial Story Generation ---")
        
        current_story = self._generate_initial_story(sentences)
        
        if self.verbose:
            print(f"📝 Initial story: {current_story[:200]}...")
        
        stories = [current_story]
        critiques = []
        
        # Step 2-4: Iterative refinement
        for iteration in range(1, self.max_refinements + 1):
            if self.verbose:
                print(f"\n--- Refinement {iteration}/{self.max_refinements} ---")
            
            # Generate critique
            critique = self._critique_story(sentences, current_story, iteration)
            critiques.append(critique)
            
            if self.verbose:
                print(f"💬 Critique: {critique[:200]}...")
            
            # Check if story is satisfactory
            if self._is_satisfactory(critique):
                if self.verbose:
                    print(f"✅ Story is satisfactory!")
                break
            
            # Refine the story
            refined_story = self._refine_story(sentences, current_story, critique)
            stories.append(refined_story)
            current_story = refined_story
            
            if self.verbose:
                print(f"🔧 Refined story: {refined_story[:200]}...")
        
        # Verify the final story
        is_valid, reason = self.verify_story(current_story, sentences)
        
        if self.verbose:
            print(f"\n  Story length: {len(current_story)} characters")
            if is_valid:
                print(f"  ✅ Valid: {reason}")
            else:
                print(f"  ⚠️  Issue: {reason}")
        
        return {
            'id': problem_id,
            'sentences': sentences,
            'story': current_story,
            'is_valid': is_valid,
            'validation_reason': reason,
            'refinements': len(stories) - 1,
            'stories': stories,
            'critiques': critiques
        }
    
    def _generate_initial_story(self, sentences: List[str]) -> str:
        """Generate initial story attempt."""
        prompt = f"""Write a 4-paragraph story.

⚠️ ABSOLUTE REQUIREMENTS:
1. Write EXACTLY 4 paragraphs (no more, no less)
2. Separate each paragraph with a blank line
3. Each paragraph MUST end with the EXACT sentence shown below (copy it word-for-word)
4. DO NOT add anything after the required sentence ends
5. DO NOT change or paraphrase the required sentences

📋 REQUIRED ENDING SENTENCES:

Paragraph 1 ends with: "{sentences[0]}"

Paragraph 2 ends with: "{sentences[1]}"

Paragraph 3 ends with: "{sentences[2]}"

Paragraph 4 ends with: "{sentences[3]}"

✍️ YOUR TASK:
- Write 3-5 sentences per paragraph
- Build each paragraph toward its required ending
- Make the required sentences fit naturally
- Connect paragraphs with a coherent narrative

⚠️ REMEMBER:
- EXACTLY 4 paragraphs
- Blank line between each paragraph
- Each paragraph ENDS with its exact required sentence
- NO text after the required ending sentence

Write the 4-paragraph story now:"""
        
        story = self.llm.complete(prompt)
        return story.strip()
    
    def _critique_story(self, sentences: List[str], story: str, iteration: int) -> str:
        """Generate critique of the current story."""
        prompt = f"""You are a critical editor reviewing a creative story. Provide detailed feedback.

Required ending sentences:
1. {sentences[0]}
2. {sentences[1]}
3. {sentences[2]}
4. {sentences[3]}

Current Story:
{story}

Critique the story on:
1. Does each paragraph end with the correct required sentence?
2. Does the story have exactly 4 paragraphs?
3. Is the narrative coherent and flows naturally?
4. Are transitions between paragraphs smooth?
5. Is it creative and engaging?
6. Are there any errors or issues?

If the story meets all requirements perfectly, start with "SATISFACTORY".
Otherwise, provide specific suggestions for improvement.

Critique:"""
        
        critique = self.llm.complete(prompt)
        return critique
    
    def _is_satisfactory(self, critique: str) -> bool:
        """Check if critique indicates story is satisfactory."""
        critique_lower = critique.lower()
        satisfactory_keywords = ['satisfactory', 'meets all requirements', 'perfect', 'excellent']
        return any(keyword in critique_lower for keyword in satisfactory_keywords)
    
    def _refine_story(self, sentences: List[str], current_story: str, critique: str) -> str:
        """Refine the story based on critique."""
        prompt = f"""Improve the following story based on the critique provided.

⚠️ ABSOLUTE REQUIREMENTS:
1. Write EXACTLY 4 paragraphs (no more, no less)
2. Separate each paragraph with a blank line
3. Each paragraph MUST end with the EXACT sentence shown below (copy it word-for-word)
4. DO NOT add anything after the required sentence ends

📋 REQUIRED ENDING SENTENCES:
Paragraph 1 ends with: "{sentences[0]}"
Paragraph 2 ends with: "{sentences[1]}"
Paragraph 3 ends with: "{sentences[2]}"
Paragraph 4 ends with: "{sentences[3]}"

Current Story:
{current_story}

Critique:
{critique}

✍️ Write an improved version that:
- Addresses all issues in the critique
- Has EXACTLY 4 paragraphs with blank lines between them
- Ends EACH paragraph with its exact required sentence (no modifications)

Improved story:"""
        
        refined_story = self.llm.complete(prompt)
        return refined_story.strip()
    
    def verify_story(self, story: str, expected_sentences: List[str]) -> tuple[bool, str]:
        """Verify the story meets requirements."""
        if not story or story == "NO_STORY_GENERATED":
            return False, "No story generated"
        
        paragraphs = [p.strip() for p in story.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 4:
            return False, f"Only {len(paragraphs)} paragraphs found (need 4)"
        
        missing_sentences = []
        for i, expected_sent in enumerate(expected_sentences):
            if i < len(paragraphs):
                normalized_para = ' '.join(paragraphs[i].split()).lower()
                normalized_sent = ' '.join(expected_sent.split()).lower()
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
                         'story', 'is_valid', 'validation_reason', 'refinements']
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
                    'refinements': result['refinements']
                })
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print statistics
        valid = sum(1 for r in results if r['is_valid'])
        print(f"\n📊 Statistics:")
        print(f"   ✓ Valid stories: {valid}/{len(results)} ({valid/len(results)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate creative stories using Self-Refine approach'
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
        '--output',
        type=str,
        default=None,
        help='Output CSV file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_name = Path(args.input_csv).stem
        output_file = f"{input_name}_selfrefine_{timestamp}.csv"
    
    print(f"\n🚀 Self-Refine Creative Writing Solver")
    print(f"{'='*80}")
    print(f"Input: {args.input_csv}")
    print(f"Output: {output_file}")
    print(f"Limit: {args.limit if args.limit else 'No limit'}")
    print(f"{'='*80}\n")
    
    # Read problems
    problems = read_sentences_from_csv(args.input_csv, limit=args.limit)
    
    if not problems:
        print("❌ No problems to process")
        return 1
    
    # Initialize solver
    solver = SelfRefineCreativeWritingSolver(verbose=not args.quiet)
    
    # Process each problem
    results = []
    
    for idx, problem in enumerate(problems, 1):
        print(f"\n🔄 Processing {idx}/{len(problems)}...")
        
        result = solver.solve(problem['sentences'], problem['id'])
        results.append(result)
    
    # Save final results
    save_results_to_csv(results, output_file)
    
    print(f"\n✅ All done! Results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
