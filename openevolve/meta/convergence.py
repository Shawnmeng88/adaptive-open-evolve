"""
Convergence analysis for meta-evolution feedback

Analyzes the convergence patterns of inner evolution runs to provide
feedback for the outer loop's prompt refinement.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Metrics describing convergence behavior of an inner evolution run"""
    
    # Primary metrics
    iterations_to_plateau: int  # When improvement stopped
    final_best_score: float
    convergence_rate: float  # iterations / score improvement (lower = faster)
    
    # Detailed analysis
    score_trajectory: List[float] = field(default_factory=list)
    improvement_iterations: List[int] = field(default_factory=list)
    
    # Pattern analysis
    stuck_patterns: List[str] = field(default_factory=list)
    successful_strategies: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    
    # Diversity metrics
    diversity_at_plateau: float = 0.0
    islands_active_at_end: int = 0
    total_valid_programs: int = 0
    total_invalid_programs: int = 0


class ConvergenceAnalyzer:
    """
    Analyzes convergence patterns from inner evolution runs
    
    Provides feedback to the Meta-LLM about what worked and what didn't,
    enabling it to generate better seed prompts.
    """
    
    def __init__(
        self,
        plateau_window: int = 20,
        plateau_threshold: float = 0.001,
    ):
        """
        Args:
            plateau_window: Number of iterations to check for plateau
            plateau_threshold: Minimum improvement to not be considered plateau
        """
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
    
    def analyze(
        self,
        convergence_trace: List[Dict[str, Any]],
        final_best_score: float,
    ) -> ConvergenceMetrics:
        """
        Analyze convergence from an evolution trace
        
        Args:
            convergence_trace: List of per-iteration data with scores
            final_best_score: The final best score achieved
            
        Returns:
            ConvergenceMetrics with detailed analysis
        """
        if not convergence_trace:
            return ConvergenceMetrics(
                iterations_to_plateau=0,
                final_best_score=final_best_score,
                convergence_rate=float('inf'),
            )
        
        # Extract score trajectory
        score_trajectory = []
        best_so_far = 0.0
        improvement_iterations = []
        
        for i, entry in enumerate(convergence_trace):
            score = entry.get('score', entry.get('combined_score', 0.0))
            if score > best_so_far:
                best_so_far = score
                improvement_iterations.append(i)
            score_trajectory.append(best_so_far)
        
        # Detect plateau
        iterations_to_plateau = self._detect_plateau(score_trajectory)
        
        # Calculate convergence rate
        if final_best_score > 0:
            convergence_rate = iterations_to_plateau / final_best_score
        else:
            convergence_rate = float('inf')
        
        # Analyze patterns
        stuck_patterns = self._extract_stuck_patterns(convergence_trace)
        successful_strategies = self._extract_successful_strategies(convergence_trace)
        failure_modes = self._extract_failure_modes(convergence_trace)
        
        # Count valid/invalid programs
        total_valid = sum(1 for e in convergence_trace if e.get('is_valid', e.get('validity', 0) > 0))
        total_invalid = len(convergence_trace) - total_valid
        
        # Calculate diversity at plateau
        diversity_at_plateau = self._calculate_diversity_at_plateau(convergence_trace, iterations_to_plateau)
        
        # Count active islands
        islands_active = len(set(e.get('island', 0) for e in convergence_trace[-min(20, len(convergence_trace)):]))
        
        return ConvergenceMetrics(
            iterations_to_plateau=iterations_to_plateau,
            final_best_score=final_best_score,
            convergence_rate=convergence_rate,
            score_trajectory=score_trajectory,
            improvement_iterations=improvement_iterations,
            stuck_patterns=stuck_patterns,
            successful_strategies=successful_strategies,
            failure_modes=failure_modes,
            diversity_at_plateau=diversity_at_plateau,
            islands_active_at_end=islands_active,
            total_valid_programs=total_valid,
            total_invalid_programs=total_invalid,
        )
    
    def _detect_plateau(self, score_trajectory: List[float]) -> int:
        """Detect when the score stopped improving"""
        if len(score_trajectory) < self.plateau_window:
            return len(score_trajectory)
        
        # Find the last iteration where meaningful improvement occurred
        for i in range(len(score_trajectory) - self.plateau_window, -1, -1):
            window_start = score_trajectory[i]
            window_end = score_trajectory[min(i + self.plateau_window, len(score_trajectory) - 1)]
            
            if window_end - window_start > self.plateau_threshold:
                return i + self.plateau_window
        
        return len(score_trajectory)
    
    def _extract_stuck_patterns(self, trace: List[Dict[str, Any]]) -> List[str]:
        """Extract patterns where evolution got stuck"""
        patterns = []
        
        # Check for repeated failures
        consecutive_failures = 0
        max_consecutive_failures = 0
        
        for entry in trace:
            score = entry.get('score', entry.get('combined_score', 0.0))
            if score == 0.0:
                consecutive_failures += 1
                max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
            else:
                consecutive_failures = 0
        
        if max_consecutive_failures >= 5:
            patterns.append(f"Repeated failures: {max_consecutive_failures} consecutive invalid programs")
        
        # Check for error patterns
        error_counts: Dict[str, int] = {}
        for entry in trace:
            error = entry.get('error', '')
            if error:
                error_key = error[:50]  # Truncate for grouping
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
        
        for error, count in error_counts.items():
            if count >= 3:
                patterns.append(f"Repeated error ({count}x): {error}")
        
        # Check for validity issues
        validity_failures = sum(1 for e in trace if e.get('validity', 1.0) == 0.0)
        if validity_failures > len(trace) * 0.5:
            patterns.append(f"High invalidity rate: {validity_failures}/{len(trace)} programs invalid")
        
        return patterns
    
    def _extract_successful_strategies(self, trace: List[Dict[str, Any]]) -> List[str]:
        """Extract patterns from successful improvements"""
        strategies = []
        
        # Find iterations where score improved
        best_score = 0.0
        improvements = []
        
        for i, entry in enumerate(trace):
            score = entry.get('score', entry.get('combined_score', 0.0))
            if score > best_score:
                improvement = score - best_score
                best_score = score
                improvements.append({
                    'iteration': i,
                    'improvement': improvement,
                    'new_score': score,
                    'changes': entry.get('changes', 'unknown'),
                })
        
        if improvements:
            strategies.append(f"Made {len(improvements)} improvements")
            
            # Analyze biggest improvements
            improvements.sort(key=lambda x: x['improvement'], reverse=True)
            for imp in improvements[:3]:
                strategies.append(
                    f"Iteration {imp['iteration']}: +{imp['improvement']:.4f} "
                    f"(changes: {imp['changes'][:50] if isinstance(imp['changes'], str) else 'multiple'})"
                )
        
        return strategies
    
    def _extract_failure_modes(self, trace: List[Dict[str, Any]]) -> List[str]:
        """Extract common failure modes"""
        modes = []
        
        # Categorize failures
        syntax_errors = 0
        runtime_errors = 0
        validity_failures = 0
        timeout_failures = 0
        
        for entry in trace:
            error = entry.get('error', '').lower()
            if 'syntax' in error:
                syntax_errors += 1
            elif 'timeout' in error:
                timeout_failures += 1
            elif error:
                runtime_errors += 1
            elif entry.get('validity', 1.0) == 0.0:
                validity_failures += 1
        
        total = len(trace)
        if syntax_errors > 0:
            modes.append(f"Syntax errors: {syntax_errors}/{total} ({100*syntax_errors/total:.1f}%)")
        if runtime_errors > 0:
            modes.append(f"Runtime errors: {runtime_errors}/{total} ({100*runtime_errors/total:.1f}%)")
        if validity_failures > 0:
            modes.append(f"Validity failures: {validity_failures}/{total} ({100*validity_failures/total:.1f}%)")
        if timeout_failures > 0:
            modes.append(f"Timeouts: {timeout_failures}/{total} ({100*timeout_failures/total:.1f}%)")
        
        return modes
    
    def _calculate_diversity_at_plateau(
        self,
        trace: List[Dict[str, Any]],
        plateau_iteration: int
    ) -> float:
        """Calculate diversity of solutions at the plateau point"""
        # Simple metric: ratio of unique scores in the last window
        if not trace:
            return 0.0
        
        window_start = max(0, plateau_iteration - self.plateau_window)
        window_entries = trace[window_start:plateau_iteration]
        
        if not window_entries:
            return 0.0
        
        scores = [e.get('score', e.get('combined_score', 0.0)) for e in window_entries]
        unique_scores = len(set(round(s, 4) for s in scores))
        
        return unique_scores / len(window_entries)
    
    def compare_runs(
        self,
        metrics_list: List[ConvergenceMetrics]
    ) -> Dict[str, Any]:
        """
        Compare multiple runs to identify trends
        
        Args:
            metrics_list: List of ConvergenceMetrics from different runs
            
        Returns:
            Dictionary with comparison statistics
        """
        if not metrics_list:
            return {}
        
        convergence_rates = [m.convergence_rate for m in metrics_list if m.convergence_rate < float('inf')]
        final_scores = [m.final_best_score for m in metrics_list]
        plateau_iters = [m.iterations_to_plateau for m in metrics_list]
        
        return {
            'num_runs': len(metrics_list),
            'avg_convergence_rate': np.mean(convergence_rates) if convergence_rates else float('inf'),
            'std_convergence_rate': np.std(convergence_rates) if convergence_rates else 0.0,
            'avg_final_score': np.mean(final_scores),
            'std_final_score': np.std(final_scores),
            'avg_plateau_iteration': np.mean(plateau_iters),
            'best_final_score': max(final_scores),
            'worst_final_score': min(final_scores),
            'improvement_trend': self._calculate_trend(final_scores),
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if values are trending up, down, or flat"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"

