
"""
Creative Writing Solver using ONLY Ollama LLM (no agent)
Reads CSV with four sentences per row and generates coherent stories.
"""
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clients.ollama_client import OllamaClient

class CreativeWritingOllama:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        load_dotenv()
        load_dotenv("config.env")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = OllamaClient(model_name=self.model, base_url=self.base_url)

    def create_prompt(self, sentences: List[str], problem_id: int = None) -> str:
        # Create a simple example to demonstrate format
        example_template = f"""EXAMPLE FORMAT (DO NOT COPY CONTENT, ONLY FOLLOW FORMAT):

In the quiet village, old Mr. Thompson tinkered with gears and springs. [Your sentence here]

The next morning brought new challenges and discoveries. [Your sentence here]

By evening, everything had changed in unexpected ways. [Your sentence here]

As night fell, the story reached its natural conclusion. [Your sentence here]
"""

        prompt = f"""You are an expert creative writer. Write a 4-paragraph story.

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

{example_template}

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

✍️ Write your 4-paragraph story now:"""
        return prompt

    def extract_story(self, llm_output: str) -> str:
        if not llm_output:
            return "NO_STORY_GENERATED"
        story = llm_output.strip()
        prefixes_to_remove = [
            "Here is the story:", "Here's the story:", "Story:", "The story:", "Output:", "Result:"
        ]
        for prefix in prefixes_to_remove:
            if story.lower().startswith(prefix.lower()):
                story = story[len(prefix):].strip()
        return story

    def verify_story(self, story: str, expected_sentences: List[str]) -> tuple[bool, str]:
        if story == "NO_STORY_GENERATED" or not story:
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

    def solve_single(self, sentences: List[str], problem_id: int = None) -> Dict[str, Any]:
        if self.verbose:
            id_str = f" [{problem_id}]" if problem_id else ""
            print(f"\n{'='*70}")
            print(f"Generating story{id_str}")
            print(f"{'='*70}")
            for i, sent in enumerate(sentences, 1):
                print(f"  {i}. {sent}")
        prompt = self.create_prompt(sentences, problem_id)
        llm_output = self.llm.complete(prompt)
        story = self.extract_story(llm_output)
        is_valid, reason = self.verify_story(story, sentences)
        if self.verbose:
            print(f"\n  Story length: {len(story)} characters")
            if is_valid:
                print(f"  ✅ Valid: {reason}")
            else:
                print(f"  ⚠️  Issue: {reason}")
            print(f"\n  Preview: {story[:200]}...")
        return {
            'id': problem_id,
            'sentences': sentences,
            'story': story,
            'is_valid': is_valid,
            'validation_reason': reason,
            'raw_output': llm_output
        }

    def solve_from_csv(self, input_csv: str, output_csv: str = None, limit: int = None) -> list[dict]:
        if output_csv is None:
            input_path = Path(input_csv)
            output_csv = input_path.parent / f"{input_path.stem}_ollama_stories.csv"
        print(f"\n{'✍️ '*30}")
        print(f"Creative Writing Solver (Ollama Only)")
        print(f"{'✍️ '*30}\n")
        print(f"Input:  {input_csv}")
        print(f"Output: {output_csv}")
        if limit:
            print(f"Solving first {limit} problems only")
        print()
        problems = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                problems.append({
                    'id': int(row['id']),
                    'sentences': [
                        row['sentence_1'],
                        row['sentence_2'],
                        row['sentence_3'],
                        row['sentence_4']
                    ]
                })
        if limit:
            problems = problems[:limit]
        print(f"Found {len(problems)} problems to solve\n")
        results = []
        success_count = 0
        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] ", end='')
            result = self.solve_single(problem['sentences'], problem['id'])
            results.append(result)
            if result['is_valid']:
                success_count += 1
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'sentence_1', 'sentence_2', 'sentence_3', 'sentence_4', 'story', 'is_valid', 'validation_reason']
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
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Total problems: {len(results)}")
        print(f"Valid stories: {success_count} ({success_count/len(results)*100:.1f}%)")
        print(f"Invalid/Issues: {len(results) - success_count}")
        print(f"\nResults saved to: {output_csv}")
        json_output = Path(output_csv).with_suffix('.json')
        results_no_raw = []
        for r in results:
            r2 = dict(r)
            r2.pop('raw_output', None)
            results_no_raw.append(r2)
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results_no_raw, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {json_output}")
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate creative stories from sentence prompts using only Ollama LLM')
    parser.add_argument('input_csv', type=str, help='Input CSV file with columns: id,sentence_1,sentence_2,sentence_3,sentence_4')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: input_name_ollama_stories.csv)')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of problems to solve (default: all)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    args = parser.parse_args()
    solver = CreativeWritingOllama(verbose=not args.quiet)
    solver.solve_from_csv(args.input_csv, output_csv=args.output, limit=args.limit)

if __name__ == '__main__':
    main()
