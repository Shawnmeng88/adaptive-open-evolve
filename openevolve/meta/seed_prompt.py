"""
Seed Prompt History Tracking

Tracks the history of seed prompts and their convergence results
to enable the Meta-LLM to learn from past attempts.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

from openevolve.meta.convergence import ConvergenceMetrics

logger = logging.getLogger(__name__)


@dataclass
class SeedPromptEntry:
    """A single entry in the seed prompt history"""
    
    seed_prompt: str
    outer_iteration: int
    convergence_metrics: ConvergenceMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Derived analysis
    was_improvement: bool = False
    key_changes_from_previous: Optional[str] = None


class SeedPromptHistory:
    """
    Tracks history of seed prompts and their results
    
    Provides analysis capabilities for the Meta-LLM to understand
    what prompt patterns led to better convergence.
    """
    
    def __init__(self):
        self.entries: List[SeedPromptEntry] = []
        self.best_convergence_rate: float = float('inf')
        self.best_prompt_index: int = -1
    
    def add(
        self,
        seed_prompt: str,
        convergence_metrics: ConvergenceMetrics,
        outer_iteration: int,
    ) -> None:
        """
        Add a new seed prompt and its results to history
        
        Args:
            seed_prompt: The seed prompt used
            convergence_metrics: Results from the inner evolution run
            outer_iteration: Which outer loop iteration this was
        """
        # Check if this was an improvement
        was_improvement = convergence_metrics.convergence_rate < self.best_convergence_rate
        
        # Analyze key changes from previous prompt
        key_changes = None
        if self.entries:
            key_changes = self._analyze_prompt_changes(
                self.entries[-1].seed_prompt,
                seed_prompt
            )
        
        entry = SeedPromptEntry(
            seed_prompt=seed_prompt,
            outer_iteration=outer_iteration,
            convergence_metrics=convergence_metrics,
            was_improvement=was_improvement,
            key_changes_from_previous=key_changes,
        )
        
        self.entries.append(entry)
        
        # Update best if improved
        if was_improvement:
            self.best_convergence_rate = convergence_metrics.convergence_rate
            self.best_prompt_index = len(self.entries) - 1
            logger.info(f"New best seed prompt at iteration {outer_iteration} "
                       f"(convergence_rate: {convergence_metrics.convergence_rate:.4f})")
    
    def get_best_prompt(self) -> Optional[str]:
        """Get the seed prompt with the best convergence rate"""
        if self.best_prompt_index >= 0:
            return self.entries[self.best_prompt_index].seed_prompt
        return None
    
    def get_successful_patterns(self) -> List[str]:
        """
        Extract patterns from prompts that led to improvements
        
        Returns:
            List of identified successful patterns
        """
        patterns = []
        
        improving_entries = [e for e in self.entries if e.was_improvement]
        
        for entry in improving_entries:
            # Extract key characteristics of successful prompts
            prompt_lower = entry.seed_prompt.lower()
            
            # Check for specific patterns
            if 'step by step' in prompt_lower or 'step-by-step' in prompt_lower:
                patterns.append("Chain-of-thought prompting improved convergence")
            
            if 'constraint' in prompt_lower:
                patterns.append("Explicit constraint mentions helped")
            
            if 'example' in prompt_lower:
                patterns.append("Including examples was beneficial")
            
            if 'avoid' in prompt_lower or 'do not' in prompt_lower:
                patterns.append("Negative guidance (what to avoid) helped")
            
            if entry.key_changes_from_previous:
                patterns.append(f"Change that helped: {entry.key_changes_from_previous}")
        
        return list(set(patterns))  # Deduplicate
    
    def get_stuck_patterns(self) -> List[str]:
        """
        Extract patterns from seed prompts where evolution got stuck
        
        Returns:
            List of identified problematic patterns
        """
        patterns = []
        
        for entry in self.entries:
            metrics = entry.convergence_metrics
            
            # Check for high invalidity
            if metrics.total_invalid_programs > metrics.total_valid_programs:
                patterns.append(
                    f"Iteration {entry.outer_iteration}: High invalidity rate - "
                    f"prompt may be too aggressive or unclear about constraints"
                )
            
            # Check for stuck patterns from convergence analysis
            for stuck in metrics.stuck_patterns:
                patterns.append(f"Iteration {entry.outer_iteration}: {stuck}")
            
            # Check for early plateau
            if metrics.iterations_to_plateau < 10 and metrics.final_best_score < 0.5:
                patterns.append(
                    f"Iteration {entry.outer_iteration}: Premature plateau at low score - "
                    f"prompt may not encourage exploration"
                )
        
        return patterns
    
    def get_slow_convergence_issues(self) -> str:
        """
        Identify issues that led to slow convergence
        
        Returns:
            Summary of slow convergence issues
        """
        issues = []
        
        for entry in self.entries:
            metrics = entry.convergence_metrics
            
            # Slow convergence (high rate)
            if metrics.convergence_rate > 1000:
                issues.append(f"Very slow convergence at iteration {entry.outer_iteration}")
            
            # Low diversity at plateau
            if metrics.diversity_at_plateau < 0.2:
                issues.append(f"Low diversity at plateau (iteration {entry.outer_iteration})")
            
            # Few active islands
            if metrics.islands_active_at_end < 2:
                issues.append(f"Most islands inactive at end (iteration {entry.outer_iteration})")
        
        if not issues:
            return "No major slow convergence issues identified"
        
        return "; ".join(issues)
    
    def format_history(self, max_entries: int = 5) -> str:
        """
        Format history for inclusion in meta-prompt
        
        Args:
            max_entries: Maximum number of entries to include
            
        Returns:
            Formatted history string
        """
        if not self.entries:
            return "No previous seed prompts tried yet."
        
        # Get most recent entries
        recent = self.entries[-max_entries:]
        
        lines = ["## Seed Prompt History (most recent first):\n"]
        
        for i, entry in enumerate(reversed(recent)):
            metrics = entry.convergence_metrics
            status = "✓ IMPROVED" if entry.was_improvement else "✗ no improvement"
            
            lines.append(f"### Attempt {entry.outer_iteration + 1} [{status}]")
            lines.append(f"- Convergence rate: {metrics.convergence_rate:.2f}")
            lines.append(f"- Final score: {metrics.final_best_score:.4f}")
            lines.append(f"- Plateau at iteration: {metrics.iterations_to_plateau}")
            lines.append(f"- Valid/Invalid programs: {metrics.total_valid_programs}/{metrics.total_invalid_programs}")
            
            if metrics.failure_modes:
                lines.append(f"- Failure modes: {', '.join(metrics.failure_modes[:2])}")
            
            # Show truncated prompt
            prompt_preview = entry.seed_prompt[:200].replace('\n', ' ')
            lines.append(f"- Prompt preview: \"{prompt_preview}...\"")
            lines.append("")
        
        # Add best prompt info
        if self.best_prompt_index >= 0:
            best = self.entries[self.best_prompt_index]
            lines.append(f"## Best Result So Far (Iteration {best.outer_iteration + 1}):")
            lines.append(f"- Convergence rate: {best.convergence_metrics.convergence_rate:.2f}")
            lines.append(f"- Final score: {best.convergence_metrics.final_best_score:.4f}")
        
        return "\n".join(lines)
    
    def _analyze_prompt_changes(self, old_prompt: str, new_prompt: str) -> str:
        """Analyze key differences between prompts"""
        old_lower = old_prompt.lower()
        new_lower = new_prompt.lower()
        
        changes = []
        
        # Check for added keywords
        keywords = ['step', 'constraint', 'avoid', 'example', 'important', 'must', 'optimize']
        for kw in keywords:
            if kw in new_lower and kw not in old_lower:
                changes.append(f"added '{kw}'")
            elif kw in old_lower and kw not in new_lower:
                changes.append(f"removed '{kw}'")
        
        # Check length change
        len_diff = len(new_prompt) - len(old_prompt)
        if abs(len_diff) > 100:
            if len_diff > 0:
                changes.append(f"expanded by {len_diff} chars")
            else:
                changes.append(f"condensed by {-len_diff} chars")
        
        return "; ".join(changes) if changes else "minor changes"
    
    def save(self, path: str) -> None:
        """Save history to JSON file"""
        data = {
            'entries': [
                {
                    'seed_prompt': e.seed_prompt,
                    'outer_iteration': e.outer_iteration,
                    'timestamp': e.timestamp,
                    'was_improvement': e.was_improvement,
                    'key_changes': e.key_changes_from_previous,
                    'metrics': {
                        'convergence_rate': e.convergence_metrics.convergence_rate,
                        'final_best_score': e.convergence_metrics.final_best_score,
                        'iterations_to_plateau': e.convergence_metrics.iterations_to_plateau,
                        'total_valid': e.convergence_metrics.total_valid_programs,
                        'total_invalid': e.convergence_metrics.total_invalid_programs,
                        'stuck_patterns': e.convergence_metrics.stuck_patterns,
                        'successful_strategies': e.convergence_metrics.successful_strategies,
                    }
                }
                for e in self.entries
            ],
            'best_convergence_rate': self.best_convergence_rate,
            'best_prompt_index': self.best_prompt_index,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved seed prompt history to {path}")
    
    def load(self, path: str) -> None:
        """Load history from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.best_convergence_rate = data['best_convergence_rate']
        self.best_prompt_index = data['best_prompt_index']
        
        self.entries = []
        for e in data['entries']:
            metrics = ConvergenceMetrics(
                convergence_rate=e['metrics']['convergence_rate'],
                final_best_score=e['metrics']['final_best_score'],
                iterations_to_plateau=e['metrics']['iterations_to_plateau'],
                total_valid_programs=e['metrics']['total_valid'],
                total_invalid_programs=e['metrics']['total_invalid'],
                stuck_patterns=e['metrics'].get('stuck_patterns', []),
                successful_strategies=e['metrics'].get('successful_strategies', []),
            )
            
            entry = SeedPromptEntry(
                seed_prompt=e['seed_prompt'],
                outer_iteration=e['outer_iteration'],
                convergence_metrics=metrics,
                timestamp=e['timestamp'],
                was_improvement=e['was_improvement'],
                key_changes_from_previous=e.get('key_changes'),
            )
            self.entries.append(entry)
        
        logger.info(f"Loaded {len(self.entries)} entries from {path}")

