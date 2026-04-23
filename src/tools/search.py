"""
Search and Expansion Tools
Handles tree expansion and web search capabilities
"""

import os
from typing import List, Dict, Any

from src.clients.llm_client import LLMClient
from src.utils.memory import ReflexionMemory


class WebSearchTool:
    """Tool for searching the web to retrieve factual information using Tavily."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ---------- ReAct call() wrapper ----------
    def call(self, query: str) -> str:
        """String-in/string-out wrapper for the ReAct executor."""
        try:
            results = self.search(query)
            if not results:
                return "No web search results found (search may be unavailable)."
            return self.extract_facts(results)
        except Exception as exc:
            return f"Web search error: {exc}"
        self.api_key = os.getenv("TAVILY_API_KEY", "")
        self.use_tavily = bool(self.api_key)
        
        if self.use_tavily:
            try:
                from tavily import TavilyClient
                self.client = TavilyClient(api_key=self.api_key)
                if self.verbose:
                    print("🔍 Web search enabled (Tavily)")
            except ImportError:
                self.use_tavily = False
                if self.verbose:
                    print("⚠️ Tavily not installed. Install with: pip install tavily-python")
                    print("   Web search will be disabled")
        elif self.verbose:
            print("⚠️ TAVILY_API_KEY not set. Web search disabled.")
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information using Tavily."""
        if not self.use_tavily:
            if self.verbose:
                print(f"⚠️ Web search unavailable (no Tavily API key)")
            return []
        
        if self.verbose:
            print(f"🔍 Searching: {query}")
        
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced"
            )
            results = []
            for item in response.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'content': item.get('content', '')
                })
            if self.verbose:
                print(f"   Found {len(results)} results")
            return results
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Tavily search error: {e}")
            return []
    
    def extract_facts(self, results: List[Dict[str, Any]]) -> str:
        """Extract and format facts from search results."""
        if not results:
            return "No information found."
        
        facts = []
        for i, result in enumerate(results, 1):
            facts.append(f"{i}. {result.get('title', 'Untitled')}")
            facts.append(f"   {result.get('content', 'No content available')}")
            if result.get('url'):
                facts.append(f"   Source: {result.get('url')}")
            facts.append("")
        
        return "\n".join(facts)


class SearchTool:
    """Tool for expanding tree nodes by generating multiple solution approaches."""
    
    def __init__(self, llm: LLMClient, memory: ReflexionMemory, verbose: bool = True):
        self.llm = llm
        self.memory = memory
        self.verbose = verbose

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Expand current node to generate K child nodes."""
        # Ensure tree exists
        tree = state.setdefault('tree', {"root": None, "nodes": {}})
        nodes_map = tree.setdefault('nodes', {})
        root = tree.get('root')
        if root is None:
            # If planning didn't create it, create minimally from problem
            root = {
                "id": "root",
                "parent_id": None,
                "depth": 0,
                "thought": state.get('plan') or state['problem'],
                "state": {"type": "problem"},
                "refinement_count": 0,
                "critique_history": [],
                "local_score": 1.0,
                "cross_score": 0.0,
                "combined_score": 0.6,
                "children": [],
                "provenance": {"method": "initialization"},
                "metadata": {"score_rationale": "Initial root node"},
            }
            tree['root'] = root
        parent_id = state.get('current_node_id') or 'root'
        parent = root if parent_id == 'root' else nodes_map.get(parent_id)
        if parent is None:
            parent = root
            parent_id = 'root'

        # If parent at max depth, do not expand
        max_depth = int(state.get('max_depth', int(os.getenv('MAX_DEPTH', '2'))))
        if int(parent.get('depth', 0)) >= max_depth:
            state['thoughts'] = []
            state['current_children_ids'] = []
            if self.verbose:
                print("\n🔎 EXPANSION PHASE (no-op: max depth reached)")
            return state

        # Expand K children from this parent
        k = int(state.get('children_per_expansion', int(os.getenv('CHILDREN_PER_EXPANSION', '3'))))
        basis = parent.get('thought') or state.get('plan') or state['problem']
        thoughts = self.llm.expand(
            node_id=parent_id,
            thought=basis,
            memory_summary=self.memory.get_summary(),
            num_children=k,
        )
        child_ids: List[str] = []

        def index_to_letters(n: int) -> str:
            # 1 -> A, 26 -> Z, 27 -> AA, etc.
            s = ""
            while n > 0:
                n, r = divmod(n - 1, 26)
                s = chr(65 + r) + s
            return s

        for i, t in enumerate(thoughts, 1):
            if parent_id == 'root':
                base_id = index_to_letters(i)  # A, B, C, ...
            else:
                base_id = f"{parent_id}{i}"  # A1, A2, B1, B2, ...
            cid = base_id
            # Ensure uniqueness if re-expanded
            if cid in nodes_map:
                suffix = 2
                while f"{base_id}_{suffix}" in nodes_map:
                    suffix += 1
                cid = f"{base_id}_{suffix}"

            nodes_map[cid] = {
                "id": cid,
                "parent_id": parent_id,
                "depth": int(parent.get('depth', 0)) + 1,
                "thought": t,
                "state": {"type": "idea"},
                "refinement_count": 0,
                "critique_history": [],
                "local_score": 0.0,
                "cross_score": 0.0,
                "combined_score": 0.0,
                "children": [],
                "provenance": {"method": "expansion"},
                "metadata": {},
            }
            child_ids.append(cid)
        parent.setdefault('children', [])
        parent['children'] = child_ids

        state['thoughts'] = thoughts
        state['current_children_ids'] = child_ids
        if thoughts:
            state['current_thought'] = thoughts[0]
            state['current_node_id'] = child_ids[0]
        if self.verbose:
            print("\n🔎 EXPANSION PHASE")
            for i, t in enumerate(thoughts, 1):
                print(f"  {i}. {t[:120]}")
        return state
