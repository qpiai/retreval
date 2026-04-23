"""
Creative Writing Solver using ReAct (Reason + Act)
ReAct interleaves reasoning and action steps to generate coherent stories.
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


class ReActCreativeWritingSolver:
    """ReAct solver for creative writing that reasons and acts iteratively."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        load_dotenv()
        load_dotenv("config.env")
        
        self.llm = LLMClient()
        
        self.max_iterations = int(os.getenv('MAX_LOOPS', '3'))
    
    def solve(self, sentences: List[str], problem_id: int = None) -> Dict[str, Any]:
        """
        Generate a creative story using ReAct approach.
        
        ReAct Pattern:
        1. Thought: Plan what to write for each paragraph
        2. Action: Write the paragraph
        3. Observation: Check if it flows and ends correctly
        4. Repeat for all 4 paragraphs
        """
        if self.verbose:
            id_str = f" [{problem_id}]" if problem_id else ""
            print(f"\n{'='*80}")
            print(f"🧠 ReAct Creative Writing Solver{id_str}")
            print(f"{'='*80}")
            for i, sent in enumerate(sentences, 1):
                print(f"  {i}. {sent}")
        
        # ReAct trace
        thoughts = []
        actions = []  # Paragraphs written
        observations = []
        
        paragraphs = []
        
        # Generate 4 paragraphs using ReAct
        for para_num in range(1, 5):
            if self.verbose:
                print(f"\n--- Paragraph {para_num}/4 ---")
            
            # Thought: Plan the paragraph
            thought = self._plan_paragraph(sentences, para_num, paragraphs, thoughts)
            thoughts.append(thought)
            
            if self.verbose:
                print(f"💭 Thought: {thought[:150]}...")
            
            # Action: Write the paragraph
            paragraph = self._write_paragraph(sentences, para_num, paragraphs, thought)
            paragraphs.append(paragraph)
            actions.append(paragraph)
            
            if self.verbose:
                print(f"✍️  Action (wrote paragraph): {paragraph[:150]}...")
            
            # Observation: Check the paragraph
            observation = self._observe_paragraph(sentences, para_num, paragraph)
            observations.append(observation)
            
            if self.verbose:
                print(f"👁️  Observation: {observation[:150]}...")
        
        # Combine paragraphs into story
        story = "\n\n".join(paragraphs)
        
        # Verify the story
        is_valid, reason = self.verify_story(story, sentences)
        
        if self.verbose:
            print(f"\n  Story length: {len(story)} characters")
            if is_valid:
                print(f"  ✅ Valid: {reason}")
            else:
                print(f"  ⚠️  Issue: {reason}")
        
        return {
            'id': problem_id,
            'sentences': sentences,
            'story': story,
            'is_valid': is_valid,
            'validation_reason': reason,
            'thoughts': thoughts,
            'actions': actions,
            'observations': observations
        }
    
    def _plan_paragraph(self, sentences: List[str], para_num: int, 
                       written_paragraphs: List[str], previous_thoughts: List[str]) -> str:
        """Generate a thought about how to write the next paragraph."""
        context = ""
        if written_paragraphs:
            context = f"\n\nStory so far:\n" + "\n\n".join(written_paragraphs)
        
        prompt = f"""You are writing a creative 4-paragraph story. Each paragraph must end with a specific sentence.

Required ending sentences:
1. {sentences[0]}
2. {sentences[1]}
3. {sentences[2]}
4. {sentences[3]}

Current task: Plan paragraph {para_num} (which must end with: "{sentences[para_num-1]}")
{context}

THOUGHT: What should paragraph {para_num} be about to naturally lead to the ending sentence? How should it connect to previous paragraphs?

Provide your planning thought:"""
        
        thought = self.llm.complete(prompt)
        return thought
    
    def _write_paragraph(self, sentences: List[str], para_num: int,
                        written_paragraphs: List[str], thought: str) -> str:
        """Write a paragraph based on the thought."""
        context = ""
        if written_paragraphs:
            context = f"\n\nStory so far:\n" + "\n\n".join(written_paragraphs)
        
        prompt = f"""You are writing paragraph {para_num} of a 4-paragraph story.

Your plan: {thought}

⚠️ CRITICAL REQUIREMENTS:
1. Write EXACTLY one paragraph (3-5 sentences)
2. The paragraph MUST end with the EXACT sentence below (word-for-word, no changes)
3. DO NOT add any text after the required ending sentence

📋 REQUIRED ENDING SENTENCES FOR ALL PARAGRAPHS:
Paragraph 1 ends with: "{sentences[0]}"
Paragraph 2 ends with: "{sentences[1]}"
Paragraph 3 ends with: "{sentences[2]}"
Paragraph 4 ends with: "{sentences[3]}"
{context}

✍️ YOUR TASK: Write paragraph {para_num} that ends EXACTLY with:
"{sentences[para_num-1]}"

⚠️ REMEMBER: Copy the ending sentence EXACTLY as shown above. No modifications!

Write paragraph {para_num} now (ONLY the paragraph, no extra text):"""
        
        paragraph = self.llm.complete(prompt)
        # Clean up the paragraph
        paragraph = paragraph.strip()
        # Remove any quotes or extra formatting
        if paragraph.startswith('"') and paragraph.endswith('"'):
            paragraph = paragraph[1:-1]
        return paragraph
    
    def _observe_paragraph(self, sentences: List[str], para_num: int, 
                          paragraph: str) -> str:
        """Observe and evaluate the written paragraph."""
        prompt = f"""Evaluate this paragraph for a creative story.

Paragraph {para_num}:
{paragraph}

Required ending: "{sentences[para_num-1]}"

OBSERVATION: 
- Does it end with the required sentence?
- Is it coherent and well-written?
- Does it flow naturally?

Provide your observation:"""
        
        observation = self.llm.complete(prompt)
        return observation
    
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
                         'story', 'is_valid', 'validation_reason']
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
                    'validation_reason': result['validation_reason']
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
        description='Generate creative stories using ReAct approach'
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
        output_file = f"{input_name}_react_{timestamp}.csv"
    
    print(f"\n🚀 ReAct Creative Writing Solver")
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
    solver = ReActCreativeWritingSolver(verbose=not args.quiet)
    
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
