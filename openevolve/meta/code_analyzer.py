"""
Code Analyzer for Meta-Evolution (LLM-Only Mode)

Analyzes generated code solutions using an LLM to provide actionable feedback.
All analysis is performed by the LLM - no regex patterns.

The "analysis LLM" can be a smaller, faster model specified via command line.
"""

import asyncio
import logging
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# Comprehensive prompt for LLM to analyze ALL code samples at once
BATCH_ANALYSIS_PROMPT = """You are a code analyst. Analyze these code samples from an evolutionary optimization run for a circle packing problem.

## CODE SAMPLES (showing {num_samples} recent iterations):
{samples_text}

## TASK
Analyze ALL the samples together and provide a comprehensive report. You must fill in EVERY section below.

Respond in this EXACT JSON format:
```json
{{
    "approaches_tried": {{
        "approach_name": {{"count": N, "success_rate": "worked/struggled/mixed", "description": "1-2 sentence description of what this approach does"}}
    }},
    "best_result": {{
        "main_idea": "2-3 sentences describing the algorithmic strategy of the best scoring code",
        "placement_method": "How circle centers are positioned",
        "radius_computation": "How radii are determined",
        "constraint_handling": "How validity is ensured",
        "score": SCORE,
        "iteration": ITER
    }},
    "score_improvements": [
        {{"iteration": N, "improvement": DELTA, "what_changed": "brief description of what code change led to improvement"}}
    ],
    "stuck_patterns": [
        "Pattern 1: Description of a recurring failure mode or dead end",
        "Pattern 2: Another stuck pattern if any"
    ],
    "novel_discoveries": [
        "Any creative or unusual technique observed (or empty if none)"
    ],
    "error_patterns": {{
        "error_type": {{"count": N, "likely_cause": "what typically causes this error"}}
    }},
    "convergence_analysis": {{
        "trend": "improving/plateauing/oscillating/declining",
        "best_score": BEST_SCORE,
        "iterations_since_improvement": N,
        "diversity": "high/medium/low (are different approaches being tried?)"
    }},
    "recommendations": [
        "Recommendation 1: Balanced, actionable suggestion",
        "Recommendation 2: Another suggestion",
        "Recommendation 3: A third suggestion (diverse from the others)"
    ]
}}
```

IMPORTANT:
- Be SPECIFIC about what each approach does, not just a category name
- For "approaches_tried", identify the ACTUAL algorithmic strategy, not just keywords
- For "stuck_patterns", identify REPEATED failures or dead-ends
- For "recommendations", give DIVERSE suggestions (not all the same direction)
- Include at least 3 approaches if different methods were tried
- Score improvements should only include SIGNIFICANT jumps (>0.01)
"""


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
    # LLM analysis result (populated after analysis)
    llm_analysis: Optional[Dict[str, Any]] = None


@dataclass 
class CodeAnalysisResult:
    """Results of LLM-based code analysis"""
    
    # Approach analysis (LLM-generated)
    approaches_tried: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Best code analysis (LLM-generated)
    best_result: Dict[str, Any] = field(default_factory=dict)
    
    # Score improvements (LLM-generated)
    score_improvements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Stuck patterns (LLM-generated)
    stuck_patterns: List[str] = field(default_factory=list)
    
    # Novel discoveries (LLM-generated)
    novel_discoveries: List[str] = field(default_factory=list)
    
    # Error patterns (LLM-generated)
    error_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Convergence analysis (LLM-generated)
    convergence_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations (LLM-generated)
    recommendations: List[str] = field(default_factory=list)
    
    # Raw LLM response for debugging
    raw_llm_response: Optional[str] = None


class CodeAnalyzer:
    """
    LLM-based code analyzer for meta-evolution.
    
    Uses a (potentially smaller/faster) LLM to analyze code samples
    and generate actionable feedback for the MetaLLM.
    
    All analysis is performed by the LLM - no regex fallback.
    """
    
    def __init__(
        self,
        analysis_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            analysis_model: LLM model for code analysis (can be smaller/faster)
            api_base: API base URL (defaults to env or OpenAI)
            api_key: API key (defaults to OPENAI_API_KEY env var)
        """
        self.samples: List[CodeSample] = []
        self.analysis_model = analysis_model
        
        # LLM client (lazy init)
        self._client: Optional[AsyncOpenAI] = None
        self._api_base = api_base
        self._api_key = api_key
        
        logger.info(f"CodeAnalyzer initialized (analysis model: {analysis_model})")
    
    def _get_client(self) -> AsyncOpenAI:
        """Lazy init for OpenAI client"""
        if self._client is None:
            api_base = self._api_base or os.environ.get("OPENAI_API_BASE")
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
            
            self._client = AsyncOpenAI(
                base_url=api_base,
                api_key=api_key,
            )
        return self._client
    
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
    
    def _format_samples_for_prompt(self, samples: List[CodeSample]) -> str:
        """Format code samples for the batch analysis prompt"""
        lines = []
        
        for i, sample in enumerate(samples):
            # Truncate code to keep prompt size reasonable
            code_preview = sample.code[:1500] if len(sample.code) > 1500 else sample.code
            
            lines.append(f"### Sample {i+1} (Iteration {sample.iteration})")
            lines.append(f"Score: {sample.score:.4f}, Validity: {sample.validity}")
            if sample.error:
                lines.append(f"Error: {sample.error[:200]}")
            lines.append("```python")
            lines.append(code_preview)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response"""
        if not content:
            return None
        
        # Try to find JSON block
        json_start = content.find('```json')
        json_end = content.rfind('```')
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start + 7:json_end].strip()
        else:
            # Try to find raw JSON
            json_start = content.find('{')
            json_end = content.rfind('}')
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end + 1]
            else:
                return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            # Try to fix common issues
            json_str = json_str.replace("'", '"')  # Single quotes to double
            try:
                return json.loads(json_str)
            except:
                return None
    
    async def analyze_async(self) -> CodeAnalysisResult:
        """
        Analyze all collected code samples using LLM.
        
        Returns:
            CodeAnalysisResult with LLM-generated analysis
        """
        result = CodeAnalysisResult()
        
        if not self.samples:
            logger.warning("No samples to analyze")
            return result
        
        # Select samples for analysis (recent + best + failures)
        sorted_by_score = sorted(self.samples, key=lambda s: s.score, reverse=True)
        sorted_by_iter = sorted(self.samples, key=lambda s: s.iteration, reverse=True)
        
        # Get: best 3 + most recent 10 + some failures
        best_samples = sorted_by_score[:3]
        recent_samples = sorted_by_iter[:10]
        failed_samples = [s for s in self.samples if s.validity == 0 or s.score < 0.1][:3]
        
        # Combine and deduplicate
        samples_to_analyze = list({id(s): s for s in best_samples + recent_samples + failed_samples}.values())
        samples_to_analyze = sorted(samples_to_analyze, key=lambda s: s.iteration)[:15]  # Cap at 15
        
        logger.info(f"Analyzing {len(samples_to_analyze)} samples with LLM ({self.analysis_model})")
        
        # Format samples for prompt
        samples_text = self._format_samples_for_prompt(samples_to_analyze)
        
        prompt = BATCH_ANALYSIS_PROMPT.format(
            num_samples=len(samples_to_analyze),
            samples_text=samples_text,
        )
        
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.analysis_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temp for consistency
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            result.raw_llm_response = content
            
            # Parse JSON from response
            parsed = self._parse_json_response(content)
            
            if parsed:
                result.approaches_tried = parsed.get('approaches_tried', {})
                result.best_result = parsed.get('best_result', {})
                result.score_improvements = parsed.get('score_improvements', [])
                result.stuck_patterns = parsed.get('stuck_patterns', [])
                result.novel_discoveries = parsed.get('novel_discoveries', [])
                result.error_patterns = parsed.get('error_patterns', {})
                result.convergence_analysis = parsed.get('convergence_analysis', {})
                result.recommendations = parsed.get('recommendations', [])
                
                logger.info(f"LLM analysis complete: {len(result.approaches_tried)} approaches, "
                           f"{len(result.recommendations)} recommendations")
            else:
                logger.warning("Failed to parse LLM response as JSON")
                result.recommendations = ["Analysis failed - could not parse LLM response"]
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            result.recommendations = [f"Analysis failed: {str(e)}"]
        
        return result
    
    def analyze(self) -> CodeAnalysisResult:
        """
        Sync wrapper for analyze_async.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.analyze_async()).result()
            else:
                return loop.run_until_complete(self.analyze_async())
        except RuntimeError:
            return asyncio.run(self.analyze_async())
    
    def format_for_prompt(self, result: Optional[CodeAnalysisResult] = None) -> str:
        """
        Format LLM analysis results for inclusion in MetaLLM prompt.
        
        All sections are LLM-generated, with structure provided here.
        """
        if result is None:
            result = self.analyze()
        
        lines = ["## Code Analysis from Recent Iterations\n"]
        lines.append("*Analysis performed by LLM*\n")
        
        # 1. Approaches Tried (LLM-generated)
        if result.approaches_tried:
            lines.append("### Approaches Tried:")
            for approach_name, info in result.approaches_tried.items():
                if isinstance(info, dict):
                    count = info.get('count', '?')
                    status = info.get('success_rate', 'unknown')
                    desc = info.get('description', '')
                    lines.append(f"  - **{approach_name}** ({count} attempts, {status})")
                    if desc:
                        lines.append(f"    {desc}")
                else:
                    lines.append(f"  - {approach_name}: {info}")
            lines.append("")
        
        # 2. Best Result Analysis (LLM-generated)
        if result.best_result:
            best = result.best_result
            score = best.get('score', '?')
            lines.append(f"### Best Result (score: {score}):")
            if best.get('main_idea'):
                lines.append(f"  **Main Idea:** {best['main_idea']}")
            if best.get('placement_method'):
                lines.append(f"  **Placement:** {best['placement_method']}")
            if best.get('radius_computation'):
                lines.append(f"  **Radius:** {best['radius_computation']}")
            if best.get('constraint_handling'):
                lines.append(f"  **Constraints:** {best['constraint_handling']}")
            lines.append("")
        
        # 3. Score Improvements (LLM-generated)
        if result.score_improvements:
            lines.append("### Score Improvements:")
            for imp in result.score_improvements[-5:]:  # Last 5
                iter_num = imp.get('iteration', '?')
                delta = imp.get('improvement', '?')
                what = imp.get('what_changed', '')
                lines.append(f"  - Iter {iter_num}: +{delta} - {what}")
            lines.append("")
        
        # 4. Stuck Patterns (LLM-generated)
        if result.stuck_patterns:
            lines.append("### Stuck Patterns (AVOID THESE):")
            for pattern in result.stuck_patterns:
                lines.append(f"  - {pattern}")
            lines.append("")
        
        # 5. Novel Discoveries (LLM-generated)
        if result.novel_discoveries:
            lines.append("### Novel Discoveries:")
            for discovery in result.novel_discoveries:
                if discovery and discovery.lower() not in ['none', 'n/a', '']:
                    lines.append(f"  - {discovery}")
            lines.append("")
        
        # 6. Error Patterns (LLM-generated)
        if result.error_patterns:
            lines.append("### Error Patterns:")
            for error_type, info in result.error_patterns.items():
                if isinstance(info, dict):
                    count = info.get('count', '?')
                    cause = info.get('likely_cause', '')
                    lines.append(f"  - {error_type}: {count} occurrences")
                    if cause:
                        lines.append(f"    Likely cause: {cause}")
                else:
                    lines.append(f"  - {error_type}: {info}")
            lines.append("")
        
        # 7. Convergence Analysis (LLM-generated)
        if result.convergence_analysis:
            conv = result.convergence_analysis
            lines.append("### Convergence Analysis:")
            if conv.get('trend'):
                lines.append(f"  - Trend: {conv['trend']}")
            if conv.get('best_score'):
                lines.append(f"  - Best Score: {conv['best_score']}")
            if conv.get('iterations_since_improvement'):
                lines.append(f"  - Iterations Since Improvement: {conv['iterations_since_improvement']}")
            if conv.get('diversity'):
                lines.append(f"  - Approach Diversity: {conv['diversity']}")
            lines.append("")
        
        # 8. Recommendations (LLM-generated)
        if result.recommendations:
            lines.append("### Recommendations:")
            for rec in result.recommendations:
                lines.append(f"  - {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all samples for a fresh analysis"""
        self.samples = []

