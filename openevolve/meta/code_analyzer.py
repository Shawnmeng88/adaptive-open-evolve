"""
Code Analyzer for Meta-Evolution

Analyzes generated code solutions to provide actionable feedback to the MetaLLM.
This enables the outer loop to understand what algorithmic approaches work.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CodeSample:
    """A code sample with its evaluation results"""
    code: str
    score: float
    validity: float
    iteration: int
    parent_id: Optional[str] = None
    error: Optional[str] = None
    changes_description: Optional[str] = None


@dataclass 
class CodeAnalysisResult:
    """Results of analyzing code samples from an evolution run"""
    
    # Approach detection
    approaches_tried: Dict[str, int] = field(default_factory=dict)  # approach -> count
    successful_approaches: List[str] = field(default_factory=list)
    failed_approaches: List[str] = field(default_factory=list)
    
    # Improvement analysis
    improvement_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Failure analysis  
    common_errors: Dict[str, int] = field(default_factory=dict)
    validity_issues: List[str] = field(default_factory=list)
    
    # Best code info
    best_code_approach: str = "unknown"
    best_code_key_features: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class CodeAnalyzer:
    """
    Analyzes generated code to extract actionable insights for the MetaLLM.
    """
    
    # Patterns to detect algorithmic approaches
    APPROACH_PATTERNS = {
        'linear_programming': [
            r'linprog', r'linear.*program', r'scipy\.optimize\.linprog',
            r'cvxpy', r'minimize.*linear'
        ],
        'quadratic_programming': [
            r'quadprog', r'qp_solve', r'minimize.*quadratic'
        ],
        'nonlinear_optimization': [
            r'minimize\(', r'scipy\.optimize\.minimize', r'SLSQP', r'BFGS',
            r'differential_evolution', r'basin_hopping'
        ],
        'grid_based': [
            r'grid', r'lattice', r'hexagonal', r'triangular.*grid',
            r'meshgrid', r'linspace.*linspace'
        ],
        'random_search': [
            r'random\.uniform', r'random\.random', r'np\.random',
            r'random.*placement', r'monte.*carlo'
        ],
        'iterative_refinement': [
            r'for.*in.*range.*:', r'while.*:', r'iterations',
            r'refine', r'improve.*loop'
        ],
        'force_directed': [
            r'force', r'repulsion', r'attraction', r'physics',
            r'spring', r'particle'
        ],
        'greedy': [
            r'greedy', r'best.*first', r'largest.*first',
            r'sorted.*reverse'
        ],
        'constraint_solving': [
            r'constraint', r'feasib', r'satisfy',
            r'check.*overlap', r'check.*boundary'
        ],
    }
    
    def __init__(self):
        self.samples: List[CodeSample] = []
    
    def add_sample(
        self,
        code: str,
        score: float,
        validity: float,
        iteration: int,
        parent_id: Optional[str] = None,
        error: Optional[str] = None,
        changes: Optional[str] = None,
    ) -> None:
        """Add a code sample for analysis"""
        self.samples.append(CodeSample(
            code=code,
            score=score,
            validity=validity,
            iteration=iteration,
            parent_id=parent_id,
            error=error,
            changes_description=changes,
        ))
    
    def analyze(self) -> CodeAnalysisResult:
        """
        Analyze all collected code samples.
        
        Returns:
            CodeAnalysisResult with detailed analysis
        """
        result = CodeAnalysisResult()
        
        if not self.samples:
            return result
        
        # Sort by score to identify best
        sorted_samples = sorted(self.samples, key=lambda s: s.score, reverse=True)
        best_sample = sorted_samples[0] if sorted_samples else None
        
        # Analyze approaches
        for sample in self.samples:
            approaches = self._detect_approaches(sample.code)
            for approach in approaches:
                result.approaches_tried[approach] = result.approaches_tried.get(approach, 0) + 1
                
                if sample.score > 0.5 and sample.validity > 0:
                    if approach not in result.successful_approaches:
                        result.successful_approaches.append(approach)
                elif sample.validity == 0 or sample.score < 0.1:
                    if approach not in result.failed_approaches and approach not in result.successful_approaches:
                        result.failed_approaches.append(approach)
        
        # Analyze improvements
        result.improvement_changes = self._analyze_improvements()
        
        # Analyze failures
        result.common_errors = self._analyze_errors()
        result.validity_issues = self._analyze_validity_issues()
        
        # Analyze best code
        if best_sample and best_sample.score > 0:
            result.best_code_approach = self._detect_primary_approach(best_sample.code)
            result.best_code_key_features = self._extract_key_features(best_sample.code)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        return result
    
    def _detect_approaches(self, code: str) -> List[str]:
        """Detect which algorithmic approaches are used in the code"""
        code_lower = code.lower()
        detected = []
        
        for approach, patterns in self.APPROACH_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code_lower):
                    detected.append(approach)
                    break
        
        return detected if detected else ['unknown']
    
    def _detect_primary_approach(self, code: str) -> str:
        """Detect the primary/dominant approach in the code"""
        approaches = self._detect_approaches(code)
        
        # Prioritize certain approaches as "primary"
        priority = ['linear_programming', 'nonlinear_optimization', 'grid_based', 
                   'force_directed', 'random_search', 'iterative_refinement']
        
        for p in priority:
            if p in approaches:
                return p
        
        return approaches[0] if approaches else 'unknown'
    
    def _extract_key_features(self, code: str) -> List[str]:
        """Extract key features/techniques from code"""
        features = []
        code_lower = code.lower()
        
        # Check for specific techniques
        if 'hexagonal' in code_lower or 'hex_' in code_lower:
            features.append("hexagonal grid layout")
        
        if 'linprog' in code_lower:
            features.append("LP for radius optimization")
        
        if 'pairwise' in code_lower or 'distance_matrix' in code_lower:
            features.append("pairwise distance constraints")
        
        if 'boundary' in code_lower or 'border' in code_lower:
            features.append("boundary constraint handling")
        
        if 'jitter' in code_lower or 'perturb' in code_lower:
            features.append("position perturbation")
        
        if 'multi_start' in code_lower or 'best_of' in code_lower:
            features.append("multi-start strategy")
        
        if 'clip' in code_lower or 'clamp' in code_lower:
            features.append("value clipping for safety")
        
        if 'safety' in code_lower or '0.999' in code_lower or '1e-6' in code_lower:
            features.append("safety margins for numerical stability")
        
        return features[:5]  # Limit to top 5
    
    def _analyze_improvements(self) -> List[Dict[str, Any]]:
        """Analyze what code changes led to score improvements"""
        improvements = []
        
        # Track best score seen so far
        best_score = 0.0
        
        for sample in sorted(self.samples, key=lambda s: s.iteration):
            if sample.score > best_score and sample.validity > 0:
                improvement = sample.score - best_score
                
                if improvement > 0.01:  # Significant improvement
                    approaches = self._detect_approaches(sample.code)
                    features = self._extract_key_features(sample.code)
                    
                    improvements.append({
                        'iteration': sample.iteration,
                        'score_before': best_score,
                        'score_after': sample.score,
                        'improvement': improvement,
                        'approaches': approaches,
                        'key_features': features,
                    })
                
                best_score = sample.score
        
        return improvements
    
    def _analyze_errors(self) -> Dict[str, int]:
        """Categorize and count errors"""
        error_counts: Dict[str, int] = {}
        
        for sample in self.samples:
            if sample.error:
                # Categorize the error
                error_lower = sample.error.lower()
                
                if 'syntax' in error_lower:
                    category = 'syntax_error'
                elif 'import' in error_lower or 'module' in error_lower:
                    category = 'import_error'
                elif 'attribute' in error_lower:
                    category = 'attribute_error'
                elif 'name' in error_lower and 'not defined' in error_lower:
                    category = 'undefined_variable'
                elif 'timeout' in error_lower:
                    category = 'timeout'
                elif 'index' in error_lower or 'bound' in error_lower:
                    category = 'index_error'
                elif 'type' in error_lower:
                    category = 'type_error'
                else:
                    category = 'other_runtime_error'
                
                error_counts[category] = error_counts.get(category, 0) + 1
        
        return error_counts
    
    def _analyze_validity_issues(self) -> List[str]:
        """Analyze what causes validity failures"""
        issues = []
        invalid_samples = [s for s in self.samples if s.validity == 0 and not s.error]
        
        if not invalid_samples:
            return issues
        
        # Sample some invalid code to analyze
        for sample in invalid_samples[:5]:
            code_lower = sample.code.lower()
            
            # Check for common validity issues
            if 'overlap' not in code_lower and 'distance' not in code_lower:
                issues.append("Missing overlap checking in code")
            
            if 'boundary' not in code_lower and 'border' not in code_lower:
                issues.append("Missing boundary constraint enforcement")
            
            if 'clip' not in code_lower and 'max(' not in code_lower and 'min(' not in code_lower:
                issues.append("No value clipping for radii")
        
        return list(set(issues))  # Deduplicate
    
    def _generate_recommendations(self, result: CodeAnalysisResult) -> List[str]:
        """Generate balanced recommendations - multiple directions, not just one"""
        recommendations = []
        
        # Identify untried approaches for diversity
        all_approaches = set(self.APPROACH_PATTERNS.keys()) - {'unknown'}
        tried_approaches = set(result.approaches_tried.keys())
        untried_approaches = all_approaches - tried_approaches
        
        # 1. What's working (fact-based)
        if result.successful_approaches:
            approaches_str = ', '.join(result.successful_approaches[:2])
            recommendations.append(f"Working approaches: {approaches_str}")
        
        # 2. What's NOT working (fact-based)
        if result.failed_approaches:
            failed_str = ', '.join(result.failed_approaches[:2])
            recommendations.append(f"Struggled with: {failed_str}")
        
        # 3. What hasn't been tried (for diversity)
        if untried_approaches:
            untried_str = ', '.join(list(untried_approaches)[:3])
            recommendations.append(f"Not yet explored: {untried_str}")
        
        # 4. Error patterns (if significant)
        if result.common_errors:
            error_summary = ', '.join(f"{k}({v})" for k, v in list(result.common_errors.items())[:2])
            recommendations.append(f"Common errors: {error_summary}")
        
        # 5. What led to the best improvement (fact)
        if result.improvement_changes:
            best_imp = max(result.improvement_changes, key=lambda x: x['improvement'])
            features = best_imp.get('key_features', [])
            if features:
                recommendations.append(f"Best improvement techniques: {', '.join(features[:3])}")
        
        # 6. Validity issues (if any)
        if result.validity_issues:
            recommendations.append(f"Validity issues observed: {result.validity_issues[0]}")
        
        return recommendations
    
    def format_for_prompt(self, result: Optional[CodeAnalysisResult] = None) -> str:
        """
        Format analysis results for inclusion in MetaLLM prompt.
        
        Args:
            result: Pre-computed analysis result, or None to compute
            
        Returns:
            Formatted string for prompt injection
        """
        if result is None:
            result = self.analyze()
        
        lines = ["## Code Analysis from Recent Iterations\n"]
        
        # All known approaches - show coverage
        all_approaches = set(self.APPROACH_PATTERNS.keys()) - {'unknown'}
        tried_approaches = set(result.approaches_tried.keys())
        untried_approaches = all_approaches - tried_approaches
        
        # Approaches tried with results
        if result.approaches_tried:
            lines.append("### Approaches Tried (with success rate):")
            for approach, count in sorted(result.approaches_tried.items(), key=lambda x: -x[1]):
                if approach in result.successful_approaches:
                    status = "worked"
                elif approach in result.failed_approaches:
                    status = "struggled"
                else:
                    status = "mixed"
                lines.append(f"  - {approach}: {count} attempts ({status})")
            lines.append("")
        
        # Approaches NOT tried yet
        if untried_approaches:
            lines.append("### Approaches Not Yet Tried:")
            for approach in sorted(untried_approaches):
                lines.append(f"  - {approach}")
            lines.append("")
        
        # Best performing code details
        if result.best_code_approach != 'unknown':
            lines.append(f"### Best Result Used: {result.best_code_approach}")
            if result.best_code_key_features:
                lines.append("Techniques in best code:")
                for feature in result.best_code_key_features:
                    lines.append(f"  - {feature}")
            lines.append("")
        
        # Score improvements timeline
        if result.improvement_changes:
            lines.append("### Score Improvements:")
            for imp in result.improvement_changes[-3:]:
                lines.append(
                    f"  - Iter {imp['iteration']}: +{imp['improvement']:.4f} "
                    f"({', '.join(imp['approaches'][:2])})"
                )
            lines.append("")
        
        # Errors encountered
        if result.common_errors:
            lines.append("### Errors Encountered:")
            for error, count in sorted(result.common_errors.items(), key=lambda x: -x[1])[:3]:
                lines.append(f"  - {error}: {count} occurrences")
            lines.append("")
        
        # Summary observations (neutral)
        if result.recommendations:
            lines.append("### Summary:")
            for rec in result.recommendations:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all samples for a fresh analysis"""
        self.samples = []

