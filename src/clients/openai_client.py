"""
OpenAI API Client for GPT models
Provides a unified interface compatible with other LLM clients (Ollama, Gemini)
"""
import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv


class OpenAIClient:
    """Client for OpenAI API - compatible with OllamaClient and GeminiClient interface."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, verbose: bool = False):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (if None, loads from .env)
            model_name: Model to use (default: gpt-4o-mini)
            verbose: Print initialization details
        """
        load_dotenv()
        
        # Load API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file")
        
        # Load model configuration
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("OpenAI package not installed. Run: pip install openai")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Load generation config from environment
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.95"))
        self.max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
        self.timeout = int(os.getenv("OPENAI_TIMEOUT", "30"))
        
        self.verbose = verbose
        
        if self.verbose:
            print(f"🤖 Using OpenAI model: {self.model_name}")
            print(f"   Temperature: {self.temperature}")
            print(f"   Max tokens: {self.max_tokens}")
    
    def complete(self, prompt: str) -> str:
        """
        Generate a completion for a prompt.
        
        Args:
            prompt: The input prompt
        
        Returns:
            Generated text response
        """
        return self.generate(prompt)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: The input prompt
            max_tokens: Override default max tokens
        
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens or self.max_tokens,
                timeout=self.timeout
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_msg = f"OpenAI API Error: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def evaluate_response(self, prompt: str, response_text: str) -> Dict[str, Any]:
        """
        Use OpenAI to evaluate a response (for benchmarking).
        
        Args:
            prompt: Original prompt
            response_text: Generated response to evaluate
        
        Returns:
            Evaluation dictionary with score and feedback
        """
        evaluation_prompt = f"""
Evaluate the following response to the given task.

TASK:
{prompt}

RESPONSE:
{response_text}

Provide evaluation in JSON format with:
- score: 0-10 rating
- correctness: 0-10
- completeness: 0-10
- feedback: brief explanation

Return ONLY valid JSON, no markdown.
"""
        
        try:
            result = self.generate(evaluation_prompt, max_tokens=500)
            
            # Try to parse JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback if JSON parsing fails
            return {
                "score": 5,
                "correctness": 5,
                "completeness": 5,
                "feedback": result[:200]
            }
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Evaluation error: {e}")
            return {
                "score": 0,
                "correctness": 0,
                "completeness": 0,
                "feedback": f"Evaluation failed: {str(e)}"
            }
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from text response.
        
        Args:
            text: Text containing JSON
        
        Returns:
            Parsed JSON dict or None
        """
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None
    
    def extract_code(self, text: str, language: str = "python") -> Optional[str]:
        """
        Extract code from response text.
        
        Args:
            text: Text containing code
            language: Programming language
        
        Returns:
            Extracted code or None
        """
        # Try markdown code blocks first
        patterns = [
            rf"```{language}\n(.*?)\n```",
            r"```\n(.*?)\n```",
            r"```(.*?)```"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code blocks, return text as-is (might be direct code)
        if language == "python" and ("def " in text or "import " in text):
            return text.strip()
        
        return None
