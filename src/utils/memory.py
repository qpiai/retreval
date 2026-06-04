"""
Memory Buffer — persistent reflexion-style memory with deduplication, gradient refinement, and consolidation.

Inspired by TextGrad (gradient-based text refinement) and GEPA (hash dedup, Pareto-efficient
candidate management). Entries are structured MemoryEntry objects with confidence tracking,
keyword-based deduplication, and cross-run persistence.
"""
from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import json
import os
try:
    import fcntl  # Unix only; absent on Windows
except ImportError:  # pragma: no cover
    fcntl = None


# ---------------------------------------------------------------------------
# Stopwords for keyword extraction (common English words to ignore)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all both each few more most "
    "other some such no nor not only own same so than too very and but if or "
    "because until while about against it its this that these those i me my we "
    "our you your he him his she her they them their what which who whom".split()
)


# ---------------------------------------------------------------------------
# MemoryEntry dataclass
# ---------------------------------------------------------------------------
@dataclass
class MemoryEntry:
    """A single structured memory entry with confidence tracking."""
    id: str                            # SHA256[:16] of normalized content
    content: str                       # The insight/failure/success text
    category: str                      # "insight" | "failure" | "success"
    created_at: str                    # ISO timestamp
    updated_at: str                    # ISO timestamp of last refinement
    version: int = 1                   # Refinement count
    confidence: float = 0.5            # 0.0-1.0
    hit_count: int = 0                 # Times used when problem succeeded
    miss_count: int = 0                # Times used when problem failed
    gradients: List[str] = field(default_factory=list)       # Last 3 critiques
    source_problems: List[str] = field(default_factory=list) # Problem IDs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class Memory:
    """
    Persistent memory system for storing reasoning insights.
    Backed by a .md file — survives across program runs with no entry limit.

    Features over the original:
    - MemoryEntry objects with confidence, hit/miss counts, gradients
    - Keyword-based deduplication (Jaccard similarity)
    - TextGrad-inspired gradient refinement on failure/success
    - Periodic consolidation (merge similar, retire low-confidence)
    - Backward-compatible load from legacy string-list format
    """

    def __init__(self, memory_file: Optional[str] = None, auto_save: bool = False,
                 max_working_size: int = 10, load_on_init: bool = False):
        if memory_file is None:
            project_root = Path(__file__).parent.parent.parent
            memory_file = str(project_root / "memory.md")
        self.memory_file = memory_file
        self.auto_save = auto_save
        self.max_working_size = max_working_size
        self.load_on_init = load_on_init

        # Structured entries (replaces plain string lists)
        self.entries: List[MemoryEntry] = []

        # Legacy lists kept as *views* for backward compat with get_summary callers
        self.best_paths: List[str] = []
        self.iteration_count: int = 0

        # Aggregate stats (unchanged from original)
        self.failure_patterns: Dict[str, int] = {}
        self.success_patterns: Dict[str, int] = {}
        self.problem_type_performance: Dict[str, List[bool]] = {}
        self.approach_reliability: Dict[str, float] = {}
        self.failure_details: List[Dict[str, Any]] = []
        self.success_details: List[Dict[str, Any]] = []

        self._dirty = False

        if self.load_on_init:
            self._load()

    # ------------------------------------------------------------------
    # Text normalization & keyword extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase, strip punctuation/whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """Extract meaningful keywords (minus stopwords)."""
        normalized = Memory._normalize_text(text)
        words = set(re.findall(r"\b[a-z]{2,}\b", normalized))
        return words - _STOPWORDS

    @staticmethod
    def _content_hash(text: str) -> str:
        """SHA256[:16] of normalized content."""
        return hashlib.sha256(Memory._normalize_text(text).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------
    def _find_similar(self, keywords: set, category: str, threshold: float = 0.6) -> Optional[MemoryEntry]:
        """Find existing entry with Jaccard similarity above threshold."""
        if not keywords:
            return None
        best_entry = None
        best_score = 0.0
        for entry in self.entries:
            if entry.category != category:
                continue
            entry_kw = self._extract_keywords(entry.content)
            if not entry_kw:
                continue
            intersection = keywords & entry_kw
            union = keywords | entry_kw
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard >= threshold and jaccard > best_score:
                best_score = jaccard
                best_entry = entry
        return best_entry

    def _merge_entries(self, existing: MemoryEntry, new_text: str, problem_id: str = "") -> None:
        """Merge new text into existing entry: keep longer/more detailed version."""
        now = datetime.now().isoformat()
        # Keep the more detailed content
        if len(new_text) > len(existing.content):
            existing.content = new_text
            existing.id = self._content_hash(new_text)
        existing.updated_at = now
        existing.version += 1
        existing.hit_count += 1
        if problem_id and problem_id not in existing.source_problems:
            existing.source_problems.append(problem_id)

    # ------------------------------------------------------------------
    # Core add methods (with dedup)
    # ------------------------------------------------------------------
    def _add_entry(self, text: str, category: str, problem_id: str = "") -> None:
        """Add an entry with deduplication."""
        if not text or not text.strip():
            return

        content_id = self._content_hash(text)

        # 1. Exact duplicate check
        for entry in self.entries:
            if entry.id == content_id and entry.category == category:
                entry.hit_count += 1
                entry.updated_at = datetime.now().isoformat()
                if problem_id and problem_id not in entry.source_problems:
                    entry.source_problems.append(problem_id)
                self._dirty = True
                return

        # 2. Semantic overlap check (Jaccard > 0.6)
        keywords = self._extract_keywords(text)
        similar = self._find_similar(keywords, category, threshold=0.6)
        if similar is not None:
            self._merge_entries(similar, text, problem_id)
            self._dirty = True
            return

        # 3. Truly novel — create new MemoryEntry
        now = datetime.now().isoformat()
        entry = MemoryEntry(
            id=content_id,
            content=text,
            category=category,
            created_at=now,
            updated_at=now,
            version=1,
            confidence=0.5,
            hit_count=1,
            miss_count=0,
            gradients=[],
            source_problems=[problem_id] if problem_id else [],
        )
        self.entries.append(entry)
        self._dirty = True

        if self.auto_save:
            self._save()

    def add_insight(self, insight: str, problem_id: str = "") -> None:
        """Add a successful reasoning insight (with dedup)."""
        self._add_entry(insight, "insight", problem_id)

    def add_failure(self, failure: str, problem_id: str = "") -> None:
        """Add a failed reasoning pattern to avoid (with dedup)."""
        self._add_entry(failure, "failure", problem_id)

    def add_success(self, success: str, problem_id: str = "") -> None:
        """Add a successful reasoning pattern (with dedup)."""
        self._add_entry(success, "success", problem_id)

    # ------------------------------------------------------------------
    # TextGrad-inspired gradient methods
    # ------------------------------------------------------------------
    def apply_gradients_on_failure(self, failure_context: str, llm_callable: Callable = None) -> None:
        """
        On failure: decrease confidence of active entries, store textual gradient.
        If llm_callable is provided and an entry accumulates 3+ gradients, refine it.
        """
        active = [e for e in self.entries if e.confidence > 0.2]
        for entry in active:
            # Decrease confidence
            entry.confidence = max(0.0, entry.confidence - 0.1)
            entry.miss_count += 1
            entry.updated_at = datetime.now().isoformat()

            # Generate textual gradient (critique)
            gradient = f"Failed with context: {failure_context[:200]}"
            entry.gradients.append(gradient)
            # Keep only last 3 gradients
            if len(entry.gradients) > 3:
                entry.gradients = entry.gradients[-3:]

            # If 3+ gradients accumulated and LLM available, refine
            if len(entry.gradients) >= 3 and llm_callable is not None:
                self._refine_entry(entry, llm_callable)

        self._dirty = True

    def apply_gradients_on_success(self) -> None:
        """On success: increase confidence and hit_count for all active entries."""
        active = [e for e in self.entries if e.confidence > 0.2]
        for entry in active:
            entry.confidence = min(1.0, entry.confidence + 0.1)
            entry.hit_count += 1
            entry.updated_at = datetime.now().isoformat()
        self._dirty = True

    def _refine_entry(self, entry: MemoryEntry, llm_callable: Callable) -> None:
        """Use LLM to rewrite entry content based on accumulated gradients."""
        gradients_text = "\n".join(f"- {g}" for g in entry.gradients)
        prompt = (
            f"Refine the following memory entry based on recent failure feedback.\n\n"
            f"Current entry ({entry.category}): {entry.content}\n\n"
            f"Recent failure gradients:\n{gradients_text}\n\n"
            f"Rewrite the entry to be more accurate and actionable. "
            f"Keep it concise (1-2 sentences). Output ONLY the refined text."
        )
        try:
            refined = llm_callable(prompt)
            if refined and len(refined.strip()) > 10:
                entry.content = refined.strip()
                entry.id = self._content_hash(entry.content)
                entry.version += 1
                entry.gradients = []  # Reset gradients after refinement
                entry.updated_at = datetime.now().isoformat()
        except Exception:
            pass  # Keep original if refinement fails

    def consolidate(self, llm_callable: Callable = None) -> None:
        """
        Periodic consolidation (call every ~5 iterations):
        1. Retire entries with confidence < 0.2
        2. Merge entries with high Jaccard overlap within same category
        3. Optionally synthesize related entries via LLM
        """
        # 1. Retire low-confidence entries
        self.entries = [e for e in self.entries if e.confidence >= 0.2]

        # 2. Merge similar entries within each category
        for category in ("insight", "failure", "success"):
            cat_entries = [e for e in self.entries if e.category == category]
            merged_ids = set()
            for i, e1 in enumerate(cat_entries):
                if e1.id in merged_ids:
                    continue
                kw1 = self._extract_keywords(e1.content)
                for e2 in cat_entries[i + 1:]:
                    if e2.id in merged_ids:
                        continue
                    kw2 = self._extract_keywords(e2.content)
                    union = kw1 | kw2
                    if not union:
                        continue
                    jaccard = len(kw1 & kw2) / len(union)
                    if jaccard >= 0.7:
                        # Merge e2 into e1
                        self._merge_entries(e1, e2.content)
                        e1.confidence = max(e1.confidence, e2.confidence)
                        e1.hit_count += e2.hit_count
                        e1.miss_count += e2.miss_count
                        for pid in e2.source_problems:
                            if pid not in e1.source_problems:
                                e1.source_problems.append(pid)
                        merged_ids.add(e2.id)

            # Remove merged entries
            if merged_ids:
                self.entries = [e for e in self.entries if e.id not in merged_ids]

        self._dirty = True

    # ------------------------------------------------------------------
    # Legacy public API (unchanged signatures)
    # ------------------------------------------------------------------
    def update_best_paths(self, path_ids: List[str]) -> None:
        self.best_paths = path_ids
        self._dirty = True
        if self.auto_save:
            self._save()

    def increment_iteration(self) -> None:
        self.iteration_count += 1
        self._dirty = True
        if self.auto_save:
            self._save()

    def record_failure(self, failure_type: str, failure_detail: Dict[str, Any]) -> None:
        self.failure_patterns[failure_type] = self.failure_patterns.get(failure_type, 0) + 1
        self.failure_details.append({
            'type': failure_type,
            'detail': failure_detail,
            'iteration': self.iteration_count
        })
        self._dirty = True
        if self.auto_save:
            self._save()

    def record_success(self, success_type: str, success_detail: Dict[str, Any]) -> None:
        self.success_patterns[success_type] = self.success_patterns.get(success_type, 0) + 1
        self.success_details.append({
            'type': success_type,
            'detail': success_detail,
            'iteration': self.iteration_count
        })
        self._dirty = True
        if self.auto_save:
            self._save()

    def record_problem_type_result(self, problem_type: str, success: bool) -> None:
        if problem_type not in self.problem_type_performance:
            self.problem_type_performance[problem_type] = []
        self.problem_type_performance[problem_type].append(success)
        self._dirty = True
        if self.auto_save:
            self._save()

    def update_approach_reliability(self, approach: str, success: bool) -> None:
        if approach not in self.approach_reliability:
            self.approach_reliability[approach] = 0.5
        current = self.approach_reliability[approach]
        success_val = 1.0 if success else 0.0
        self.approach_reliability[approach] = 0.8 * current + 0.2 * success_val
        self._dirty = True
        if self.auto_save:
            self._save()

    def save(self) -> None:
        """Explicitly save to disk (only if there are changes)."""
        if self._dirty:
            self._save()
            self._dirty = False

    def get_approach_skepticism(self, approach: str) -> float:
        if approach not in self.approach_reliability:
            return 0.0
        reliability = self.approach_reliability.get(approach, 0.5)
        return max(0.0, 1.0 - reliability) * 0.4

    # ------------------------------------------------------------------
    # get_summary — confidence-weighted, prioritized output
    # ------------------------------------------------------------------
    def get_summary(self) -> str:
        """
        Confidence-weighted, prioritized summary.
        Filters entries with confidence > 0.3, sorts by confidence * hit_count,
        tags with [HIGH]/[MED]/[LOW], caps at 5+5+3.
        """
        def _tag(conf: float) -> str:
            if conf >= 0.7:
                return "[HIGH]"
            elif conf >= 0.5:
                return "[MED]"
            return "[LOW]"

        def _sort_key(e: MemoryEntry) -> float:
            return e.confidence * max(e.hit_count, 1)

        active = [e for e in self.entries if e.confidence > 0.3]
        insights = sorted([e for e in active if e.category == "insight"], key=_sort_key, reverse=True)[:5]
        failures = sorted([e for e in active if e.category == "failure"], key=_sort_key, reverse=True)[:5]
        successes = sorted([e for e in active if e.category == "success"], key=_sort_key, reverse=True)[:3]

        summary: List[str] = []

        if insights:
            summary.append("INSIGHTS:")
            for i, e in enumerate(insights, 1):
                summary.append(f"  {i}. {_tag(e.confidence)} {e.content}")

        if successes:
            summary.append("\nSUCCESSFUL PATTERNS:")
            for i, e in enumerate(successes, 1):
                summary.append(f"  {i}. {_tag(e.confidence)} {e.content}")

        if failures:
            summary.append("\nFAILURES TO AVOID:")
            for i, e in enumerate(failures, 1):
                summary.append(f"  {i}. {_tag(e.confidence)} {e.content}")

        if self.best_paths:
            summary.append(f"\nBEST PATHS: {', '.join(self.best_paths)}")

        return "\n".join(summary) if summary else "No memory insights yet."

    # ------------------------------------------------------------------
    # get_relevant — query-based retrieval (for KV cache efficiency)
    # ------------------------------------------------------------------
    def get_relevant(self, query: str, max_entries: int = 3) -> str:
        """Retrieve only memory entries relevant to the query.

        Instead of dumping all entries via get_summary(), this uses keyword
        overlap to find the most relevant entries. This keeps prompts smaller
        so the shared prefix stays stable for vLLM prefix caching.

        Args:
            query: The current thought/problem text to match against.
            max_entries: Maximum number of entries to return.

        Returns:
            Formatted string of relevant entries, or empty string if none.
        """
        if not query or not self.entries:
            return ""

        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            # Fallback to truncated summary
            summary = self.get_summary()
            return summary[:300] if summary != "No memory insights yet." else ""

        # Score each active entry by keyword overlap with query
        scored: List[tuple] = []
        for entry in self.entries:
            if entry.confidence <= 0.3:
                continue
            entry_keywords = self._extract_keywords(entry.content)
            if not entry_keywords:
                continue
            overlap = len(query_keywords & entry_keywords)
            if overlap > 0:
                relevance = overlap * entry.confidence
                scored.append((relevance, entry))

        if not scored:
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_entries]

        def _tag(conf: float) -> str:
            if conf >= 0.7:
                return "[HIGH]"
            elif conf >= 0.5:
                return "[MED]"
            return "[LOW]"

        lines = []
        for _, entry in top:
            lines.append(f"- {_tag(entry.confidence)} ({entry.category}) {entry.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience properties for backward compat
    # ------------------------------------------------------------------
    @property
    def insights(self) -> List[str]:
        return [e.content for e in self.entries if e.category == "insight"]

    @property
    def failures(self) -> List[str]:
        return [e.content for e in self.entries if e.category == "failure"]

    @property
    def successes(self) -> List[str]:
        return [e.content for e in self.entries if e.category == "success"]

    # ------------------------------------------------------------------
    # File I/O — with backward compatibility
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load memory from .md file. Auto-detects legacy vs new format."""
        if not os.path.exists(self.memory_file):
            return
        try:
            with open(self.memory_file, 'r') as f:
                content = f.read()
            marker_start = "<!-- JSON_DATA_START -->"
            marker_end = "<!-- JSON_DATA_END -->"
            idx_s = content.find(marker_start)
            idx_e = content.find(marker_end)
            if idx_s == -1 or idx_e == -1:
                return
            json_str = content[idx_s + len(marker_start):idx_e].strip()
            data = json.loads(json_str)

            # Load aggregate stats
            self.failure_patterns = data.get("failure_patterns", {})
            self.success_patterns = data.get("success_patterns", {})
            self.problem_type_performance = data.get("problem_type_performance", {})
            self.approach_reliability = data.get("approach_reliability", {})
            self.failure_details = data.get("failure_details", [])
            self.success_details = data.get("success_details", [])
            self.iteration_count = data.get("iteration_count", 0)
            self.best_paths = data.get("best_paths", [])

            # --- Detect format ---
            if "entries" in data and isinstance(data["entries"], list):
                # New format: structured MemoryEntry dicts
                for d in data["entries"][-self.max_working_size * 3:]:
                    try:
                        self.entries.append(MemoryEntry.from_dict(d))
                    except Exception:
                        continue
            else:
                # Legacy format: plain string lists → auto-convert
                now = datetime.now().isoformat()
                for text in data.get("insights", [])[-self.max_working_size:]:
                    self.entries.append(MemoryEntry(
                        id=self._content_hash(text), content=text, category="insight",
                        created_at=now, updated_at=now,
                    ))
                for text in data.get("failures", [])[-self.max_working_size:]:
                    self.entries.append(MemoryEntry(
                        id=self._content_hash(text), content=text, category="failure",
                        created_at=now, updated_at=now,
                    ))
                for text in data.get("successes", [])[-self.max_working_size:]:
                    self.entries.append(MemoryEntry(
                        id=self._content_hash(text), content=text, category="success",
                        created_at=now, updated_at=now,
                    ))

        except (json.JSONDecodeError, IOError):
            pass  # Start fresh if corrupted

    def _save(self) -> None:
        """Persist current memory state to .md file with confidence scores."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: List[str] = []
        lines.append("# ReTreVal Memory")
        lines.append(f"\n_Last updated: {now}_\n")

        def _tag(conf: float) -> str:
            if conf >= 0.7:
                return "[HIGH]"
            elif conf >= 0.5:
                return "[MED]"
            return "[LOW]"

        # --- Insights ---
        insight_entries = [e for e in self.entries if e.category == "insight"]
        lines.append("## Insights")
        if insight_entries:
            for i, e in enumerate(insight_entries, 1):
                lines.append(f"{i}. {_tag(e.confidence)} (v{e.version}, conf={e.confidence:.2f}) {e.content}")
        else:
            lines.append("_No insights yet._")

        # --- Successes ---
        success_entries = [e for e in self.entries if e.category == "success"]
        lines.append("\n## Successes")
        if success_entries:
            for i, e in enumerate(success_entries, 1):
                lines.append(f"{i}. {_tag(e.confidence)} (v{e.version}, conf={e.confidence:.2f}) {e.content}")
        else:
            lines.append("_No successes recorded yet._")

        # --- Failures ---
        failure_entries = [e for e in self.entries if e.category == "failure"]
        lines.append("\n## Failures to Avoid")
        if failure_entries:
            for i, e in enumerate(failure_entries, 1):
                lines.append(f"{i}. {_tag(e.confidence)} (v{e.version}, conf={e.confidence:.2f}) {e.content}")
        else:
            lines.append("_No failures recorded yet._")

        # --- Best Paths ---
        lines.append("\n## Best Paths")
        if self.best_paths:
            lines.append(", ".join(self.best_paths))
        else:
            lines.append("_None yet._")

        # --- Approach Reliability ---
        lines.append("\n## Approach Reliability")
        if self.approach_reliability:
            for approach, score in sorted(self.approach_reliability.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- **{approach}**: {score:.2f}")
        else:
            lines.append("_No approach data yet._")

        # --- Problem Type Performance ---
        lines.append("\n## Problem Type Performance")
        if self.problem_type_performance:
            for ptype, results in self.problem_type_performance.items():
                total = len(results)
                wins = sum(results)
                rate = (wins / total * 100) if total > 0 else 0
                lines.append(f"- **{ptype}**: {wins}/{total} ({rate:.0f}%)")
        else:
            lines.append("_No problem type data yet._")

        # --- Detailed Success Records ---
        lines.append("\n## Detailed Success Records")
        if self.success_details:
            for rec in self.success_details:
                pid = rec.get('detail', {}).get('problem_id', '?')
                stype = rec.get('type', 'unknown')
                approach = rec.get('detail', {}).get('approach', '?')
                lines.append(f"- Problem `{pid}` | type: {stype} | approach: {approach}")
        else:
            lines.append("_No detailed success records yet._")

        # --- Detailed Failure Records ---
        lines.append("\n## Detailed Failure Records")
        if self.failure_details:
            for rec in self.failure_details:
                pid = rec.get('detail', {}).get('problem_id', '?')
                ftype = rec.get('type', 'unknown')
                approach = rec.get('detail', {}).get('approach', '?')
                lines.append(f"- Problem `{pid}` | type: {ftype} | approach: {approach}")
        else:
            lines.append("_No detailed failure records yet._")

        # --- Stats ---
        lines.append(f"\n## Stats")
        lines.append(f"- Total iterations: {self.iteration_count}")
        lines.append(f"- Total entries: {len(self.entries)}")
        lines.append(f"- Insights: {len(insight_entries)}")
        lines.append(f"- Successes: {len(success_entries)}")
        lines.append(f"- Failures: {len(failure_entries)}")
        lines.append(f"- Detailed success records: {len(self.success_details)}")
        lines.append(f"- Detailed failure records: {len(self.failure_details)}")

        # --- Hidden JSON block for structured reload ---
        data = {
            "entries": [e.to_dict() for e in self.entries],
            "best_paths": self.best_paths,
            "iteration_count": self.iteration_count,
            "failure_patterns": self.failure_patterns,
            "success_patterns": self.success_patterns,
            "problem_type_performance": self.problem_type_performance,
            "approach_reliability": self.approach_reliability,
            "failure_details": self.failure_details,
            "success_details": self.success_details,
        }
        lines.append("\n<!-- JSON_DATA_START -->")
        lines.append(json.dumps(data, indent=2))
        lines.append("<!-- JSON_DATA_END -->")

        md_content = "\n".join(lines) + "\n"

        # Atomic write with file lock
        tmp_path = self.memory_file + ".tmp"
        with open(tmp_path, 'w') as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(md_content)
            f.flush()
            os.fsync(f.fileno())
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        os.replace(tmp_path, self.memory_file)

    # ------------------------------------------------------------------
    # Remaining public API
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": len(self.entries),
            "insights": self.insights,
            "failures": self.failures,
            "successes": self.successes,
            "best_paths": self.best_paths,
            "iterations": self.iteration_count,
            "failure_patterns": self.failure_patterns,
            "success_patterns": self.success_patterns,
            "problem_type_performance": self.problem_type_performance,
            "approach_reliability": self.approach_reliability,
            "total_failures": len(self.failure_details),
            "total_successes": len(self.success_details),
        }

    def clear(self) -> None:
        """Clear all memory (both in-memory and file)."""
        self.entries.clear()
        self.best_paths.clear()
        self.iteration_count = 0
        self.failure_patterns.clear()
        self.success_patterns.clear()
        self.problem_type_performance.clear()
        self.approach_reliability.clear()
        self.failure_details.clear()
        self.success_details.clear()
        self._dirty = True
        self._save()

    def __del__(self):
        if hasattr(self, '_dirty') and self._dirty:
            try:
                self._save()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (f"Memory(entries={len(self.entries)}, "
                f"iterations={self.iteration_count}, "
                f"file={self.memory_file})")
