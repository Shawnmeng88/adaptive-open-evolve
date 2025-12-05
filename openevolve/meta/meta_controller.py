"""
Meta-Evolution Controller: Double-Nested Loop Framework

This implements the outer loop that learns and adapts seed prompts based on
inner evolution convergence feedback.

Architecture:
    Outer Loop (this controller):
        1. Generate/refine seed prompt using Meta-LLM
        2. Run inner OpenEvolve with current seed prompt
        3. Analyze convergence metrics
        4. Update prompt history
        5. Check stopping conditions
        6. Repeat
    
    Inner Loop (standard OpenEvolve):
        - Uses the seed prompt as system message
        - Runs for N iterations
        - Returns best program and convergence trace
"""

import asyncio
import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openevolve.controller import OpenEvolve
from openevolve.config import Config
from openevolve.database import Program
from openevolve.meta.convergence import ConvergenceAnalyzer, ConvergenceMetrics
from openevolve.meta.meta_llm import MetaLLM
from openevolve.meta.seed_prompt import SeedPromptHistory

logger = logging.getLogger(__name__)


@dataclass
class MetaEvolutionConfig:
    """Configuration for meta-evolution"""
    
    # Outer loop settings
    max_outer_iterations: int = 10
    inner_iterations_per_outer: int = 100
    
    # Convergence detection
    plateau_window: int = 20  # Iterations to detect plateau
    plateau_threshold: float = 0.001  # Min improvement to not be plateau
    
    # Early stopping for outer loop
    target_score: Optional[float] = None  # Stop if this score is reached
    target_convergence_rate: Optional[float] = None  # Stop if this rate is achieved
    outer_patience: int = 3  # Outer iterations without improvement before stopping
    disable_early_stopping: bool = True  # If True, always run all outer iterations
    
    # Meta-LLM settings
    meta_llm_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    meta_llm_temperature: float = 0.7
    meta_llm_api_base: Optional[str] = None  # Defaults to same as inner LLM
    
    # Literature baseline for comparison
    literature_baseline_iterations: Optional[int] = None
    literature_baseline_score: Optional[float] = None
    
    # Output settings
    save_all_prompts: bool = True
    save_convergence_traces: bool = True


@dataclass
class OuterIterationResult:
    """Result of a single outer loop iteration"""
    
    outer_iteration: int
    seed_prompt: str
    convergence_metrics: ConvergenceMetrics
    best_program: Optional[Program]
    best_score: float
    wall_clock_time: float
    inner_iterations_run: int


@dataclass
class MetaEvolutionResult:
    """Final result of meta-evolution"""
    
    best_seed_prompt: str
    best_program: Optional[Program]
    best_score: float
    best_convergence_rate: float
    
    total_inner_iterations: int
    total_outer_iterations: int
    total_wall_clock_time: float
    
    iteration_history: List[OuterIterationResult] = field(default_factory=list)
    comparison_to_baseline: Optional[Dict[str, float]] = None
    
    def summary(self) -> str:
        """Generate a summary string"""
        lines = [
            "=" * 60,
            "META-EVOLUTION RESULTS",
            "=" * 60,
            f"Total outer iterations: {self.total_outer_iterations}",
            f"Total inner iterations: {self.total_inner_iterations}",
            f"Total wall clock time: {self.total_wall_clock_time:.1f}s",
            "",
            f"Best score achieved: {self.best_score:.4f}",
            f"Best convergence rate: {self.best_convergence_rate:.2f}",
            "",
        ]
        
        if self.comparison_to_baseline:
            lines.extend([
                "Comparison to Literature Baseline:",
                f"  Our convergence rate: {self.comparison_to_baseline['our_convergence_rate']:.2f}",
                f"  Baseline rate: {self.comparison_to_baseline['baseline_convergence_rate']:.2f}",
                f"  Improvement factor: {self.comparison_to_baseline['improvement_factor']:.2f}x",
                "",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)


class MetaEvolutionController:
    """
    Outer loop controller that learns and adapts seed prompts
    
    This implements the double-nested loop framework:
    - Outer loop: Meta-LLM generates/refines seed prompts based on feedback
    - Inner loop: Standard OpenEvolve runs with current seed prompt
    """
    
    def __init__(
        self,
        base_config: Config,
        initial_program_path: str,
        evaluation_file: str,
        meta_config: MetaEvolutionConfig,
        output_dir: str = "meta_evolution_output",
    ):
        """
        Args:
            base_config: Base OpenEvolve configuration
            initial_program_path: Path to initial program
            evaluation_file: Path to evaluator
            meta_config: Meta-evolution configuration
            output_dir: Directory for outputs
        """
        self.base_config = base_config
        self.initial_program_path = initial_program_path
        self.evaluation_file = evaluation_file
        self.meta_config = meta_config
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        api_base = meta_config.meta_llm_api_base or base_config.llm.api_base
        self.meta_llm = MetaLLM(
            model=meta_config.meta_llm_model,
            temperature=meta_config.meta_llm_temperature,
            api_base=api_base,
        )
        
        self.convergence_analyzer = ConvergenceAnalyzer(
            plateau_window=meta_config.plateau_window,
            plateau_threshold=meta_config.plateau_threshold,
        )
        
        self.seed_prompt_history = SeedPromptHistory()
        
        # Tracking
        self.iteration_results: List[OuterIterationResult] = []
        self.best_convergence_rate = float('inf')
        self.best_score = 0.0
        self.best_seed_prompt: Optional[str] = None
        self.best_program: Optional[Program] = None
        self.patience_counter = 0
        self.cumulative_iterations = 0  # Track total inner iterations across outer iters
        
        # Load initial program for analysis
        with open(initial_program_path, 'r') as f:
            self.initial_code = f.read()
        
        # Create SINGLE OpenEvolve instance that persists across all outer iterations
        # This preserves islands, MAP-Elites grid, and all accumulated programs
        self.controller = OpenEvolve(
            initial_program_path=initial_program_path,
            evaluation_file=evaluation_file,
            config=base_config,
            output_dir=os.path.join(output_dir, "evolution"),
        )
        
        logger.info(f"Initialized MetaEvolutionController")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Max outer iterations: {meta_config.max_outer_iterations}")
        logger.info(f"  Inner iterations per outer: {meta_config.inner_iterations_per_outer}")
        logger.info(f"  ✓ Single OpenEvolve instance - islands & MAP-Elites persist across prompt changes")
    
    async def run(self) -> MetaEvolutionResult:
        """
        Run the meta-evolution process
        
        Returns:
            MetaEvolutionResult with best seed prompt and convergence comparison
        """
        total_start_time = time.time()
        total_inner_iterations = 0
        
        logger.info("=" * 80)
        logger.info("STARTING META-EVOLUTION")
        logger.info(f"Max outer iterations: {self.meta_config.max_outer_iterations}")
        logger.info(f"Inner iterations per outer: {self.meta_config.inner_iterations_per_outer}")
        logger.info("=" * 80)
        
        for outer_iter in range(self.meta_config.max_outer_iterations):
            logger.info(f"\n{'='*80}")
            logger.info(f"OUTER ITERATION {outer_iter + 1}/{self.meta_config.max_outer_iterations}")
            logger.info("=" * 80)
            
            # Step 1: Generate or refine seed prompt
            if outer_iter == 0:
                seed_prompt = await self._generate_initial_seed_prompt()
            else:
                seed_prompt = await self._refine_seed_prompt()
            
            # Save prompt if configured
            if self.meta_config.save_all_prompts:
                self._save_prompt(seed_prompt, outer_iter)
            
            logger.info(f"Seed prompt preview ({len(seed_prompt)} chars):")
            logger.info(f"  {seed_prompt[:200]}...")
            
            # Step 2: Run inner OpenEvolve loop
            inner_start_time = time.time()
            best_program, convergence_trace = await self._run_inner_evolution(
                seed_prompt, outer_iter
            )
            inner_wall_time = time.time() - inner_start_time
            
            # Step 3: Analyze convergence
            best_inner_score = 0.0
            if best_program and best_program.metrics:
                best_inner_score = best_program.metrics.get('combined_score', 0.0)
            
            convergence_metrics = self.convergence_analyzer.analyze(
                convergence_trace,
                best_inner_score,
            )
            
            logger.info(f"\nConvergence Analysis:")
            logger.info(f"  Iterations to plateau: {convergence_metrics.iterations_to_plateau}")
            logger.info(f"  Final best score: {convergence_metrics.final_best_score:.4f}")
            logger.info(f"  Convergence rate: {convergence_metrics.convergence_rate:.2f}")
            logger.info(f"  Valid/Invalid: {convergence_metrics.total_valid_programs}/{convergence_metrics.total_invalid_programs}")
            
            if convergence_metrics.stuck_patterns:
                logger.info(f"  Stuck patterns: {convergence_metrics.stuck_patterns[:2]}")
            
            # Step 4: Record result
            outer_result = OuterIterationResult(
                outer_iteration=outer_iter,
                seed_prompt=seed_prompt,
                convergence_metrics=convergence_metrics,
                best_program=best_program,
                best_score=best_inner_score,
                wall_clock_time=inner_wall_time,
                inner_iterations_run=self.meta_config.inner_iterations_per_outer,
            )
            self.iteration_results.append(outer_result)
            total_inner_iterations += self.meta_config.inner_iterations_per_outer
            
            # Step 5: Update seed prompt history
            self.seed_prompt_history.add(
                seed_prompt=seed_prompt,
                convergence_metrics=convergence_metrics,
                outer_iteration=outer_iter,
            )
            
            # Step 6: Check if this is the best so far
            improved = False
            
            # Check score improvement
            if best_inner_score > self.best_score:
                self.best_score = best_inner_score
                self.best_program = best_program
                improved = True
                logger.info(f"🎉 New best score: {self.best_score:.4f}")
            
            # Check convergence rate improvement
            if convergence_metrics.convergence_rate < self.best_convergence_rate:
                self.best_convergence_rate = convergence_metrics.convergence_rate
                self.best_seed_prompt = seed_prompt
                improved = True
                logger.info(f"🎉 New best convergence rate: {self.best_convergence_rate:.2f}")
            
            if improved:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.info(f"No improvement (patience: {self.patience_counter}/{self.meta_config.outer_patience})")
            
            # Save history after each iteration
            self.seed_prompt_history.save(os.path.join(self.output_dir, "prompt_history.json"))
            
            # Step 7: Check stopping conditions
            if self._should_stop(convergence_metrics):
                logger.info("Stopping outer loop early")
                break
        
        # Compile final results
        total_wall_time = time.time() - total_start_time
        
        # Calculate comparison to baseline
        comparison = None
        if (self.meta_config.literature_baseline_iterations is not None and
            self.meta_config.literature_baseline_score is not None):
            comparison = self._compare_to_baseline()
        
        result = MetaEvolutionResult(
            best_seed_prompt=self.best_seed_prompt or "",
            best_program=self.best_program,
            best_score=self.best_score,
            best_convergence_rate=self.best_convergence_rate,
            total_inner_iterations=total_inner_iterations,
            total_outer_iterations=len(self.iteration_results),
            total_wall_clock_time=total_wall_time,
            iteration_history=self.iteration_results,
            comparison_to_baseline=comparison,
        )
        
        # Save final results
        self._save_results(result)
        
        logger.info("\n" + result.summary())
        
        return result
    
    async def _generate_initial_seed_prompt(self) -> str:
        """Generate the initial seed prompt using meta-LLM"""
        problem_description = self._extract_problem_description()
        evaluation_criteria = self._extract_evaluation_criteria()
        format_requirements = self._extract_format_requirements()
        
        # Generate the creative part from MetaLLM
        creative_prompt = await self.meta_llm.generate_initial_seed_prompt(
            problem_description=problem_description,
            initial_code=self.initial_code,
            evaluation_criteria=evaluation_criteria,
        )
        
        # Prepend critical format requirements that MUST NOT be ignored
        format_preamble = f"""## CRITICAL FORMAT REQUIREMENTS (DO NOT VIOLATE)
{format_requirements}

Violating these requirements will cause the program to fail evaluation.

---

"""
        return format_preamble + creative_prompt
    
    async def _refine_seed_prompt(self) -> str:
        """Refine the seed prompt based on convergence feedback"""
        problem_description = self._extract_problem_description()
        format_requirements = self._extract_format_requirements()
        
        current_best = self.best_seed_prompt or self.seed_prompt_history.entries[-1].seed_prompt
        
        # Get the best program code to summarize for the MetaLLM
        best_program_code = ""
        if self.best_program:
            best_program_code = self.best_program.code
        elif self.initial_code:
            best_program_code = self.initial_code
        
        # Generate refined creative prompt
        refined_prompt = await self.meta_llm.refine_seed_prompt(
            seed_prompt_history=self.seed_prompt_history,
            problem_description=problem_description,
            current_best_prompt=current_best,
            best_program_code=best_program_code,
        )
        
        # Prepend critical format requirements
        format_preamble = f"""## CRITICAL FORMAT REQUIREMENTS (DO NOT VIOLATE)
{format_requirements}

Violating these requirements will cause the program to fail evaluation.

---

"""
        return format_preamble + refined_prompt
    
    async def _run_inner_evolution(
        self,
        seed_prompt: str,
        outer_iter: int,
    ) -> tuple:
        """
        Run the inner OpenEvolve loop with the given seed prompt.
        
        IMPORTANT: Reuses the SAME OpenEvolve controller, preserving:
        - All islands and their populations
        - The MAP-Elites grid and all programs in it
        - The best program found so far
        
        Only the system prompt is changed between outer iterations.
        
        Args:
            seed_prompt: The seed prompt to use as system message
            outer_iter: Current outer iteration number
            
        Returns:
            Tuple of (best_program, convergence_trace)
        """
        # Update the system prompt in the existing controller's prompt sampler
        self.controller.prompt_sampler.set_system_message(seed_prompt)
        self.controller.config.prompt.system_message = seed_prompt
        
        # Calculate iteration range
        start_iter = self.cumulative_iterations
        end_iter = start_iter + self.meta_config.inner_iterations_per_outer
        
        # Get current stats
        num_programs = len(self.controller.database.programs)
        best_prog = self.controller.database.get_best_program()
        best_score = best_prog.metrics.get('combined_score', 0) if best_prog and best_prog.metrics else 0
        
        logger.info(f"Starting inner evolution run {outer_iter}...")
        logger.info(f"  Iterations: {start_iter + 1} to {end_iter}")
        logger.info(f"  Current programs in database: {num_programs}")
        logger.info(f"  Current best score: {best_score:.4f}")
        
        # Record programs before this batch (to extract trace for just this outer iter)
        programs_before = set(self.controller.database.programs.keys())
        
        # Run evolution (controller continues from where it left off)
        best_program = await self.controller.run(
            iterations=self.meta_config.inner_iterations_per_outer
        )
        
        # Update cumulative count
        self.cumulative_iterations = end_iter
        
        # Extract trace for ONLY new programs from this outer iteration
        convergence_trace = self._extract_convergence_trace_incremental(
            self.controller.database, programs_before, start_iter
        )
        
        # Save trace if configured
        if self.meta_config.save_convergence_traces:
            inner_output_dir = os.path.join(self.output_dir, f"outer_iter_{outer_iter}")
            os.makedirs(inner_output_dir, exist_ok=True)
            trace_path = os.path.join(inner_output_dir, "convergence_trace.json")
            with open(trace_path, 'w') as f:
                json.dump(convergence_trace, f, indent=2)
        
        return best_program, convergence_trace
    
    def _extract_convergence_trace_incremental(
        self, database, programs_before: set, start_iter: int
    ) -> List[Dict[str, Any]]:
        """Extract convergence trace for only new programs added in this outer iteration"""
        trace = []
        
        # Get only NEW programs
        new_programs = [
            p for pid, p in database.programs.items() 
            if pid not in programs_before
        ]
        new_programs.sort(key=lambda p: p.iteration_found)
        
        # Get best score at start of this batch
        best_score_so_far = 0.0
        for pid in programs_before:
            prog = database.programs.get(pid)
            if prog and prog.metrics:
                score = prog.metrics.get('combined_score', 0.0)
                best_score_so_far = max(best_score_so_far, score)
        
        for prog in new_programs:
            score = prog.metrics.get('combined_score', 0.0)
            is_improvement = score > best_score_so_far
            if is_improvement:
                best_score_so_far = score
            
            trace.append({
                'iteration': prog.iteration_found,
                'score': score,
                'best_so_far': best_score_so_far,
                'validity': prog.metrics.get('validity', 0.0),
                'is_improvement': is_improvement,
            })
        
        return trace
    
    def _extract_convergence_trace(self, database) -> List[Dict[str, Any]]:
        """Extract iteration-by-iteration convergence data from database"""
        trace = []
        
        # Get programs sorted by iteration
        programs_by_iter = sorted(
            database.programs.values(),
            key=lambda p: p.iteration_found
        )
        
        best_score_so_far = 0.0
        
        for prog in programs_by_iter:
            score = prog.metrics.get('combined_score', 0.0)
            validity = prog.metrics.get('validity', 1.0)
            is_improvement = score > best_score_so_far
            
            if score > best_score_so_far:
                best_score_so_far = score
            
            trace.append({
                'iteration': prog.iteration_found,
                'score': score,
                'combined_score': score,
                'validity': validity,
                'is_valid': validity > 0,
                'best_so_far': best_score_so_far,
                'is_improvement': is_improvement,
                'parent_id': prog.parent_id,
                'island': prog.metadata.get('island', 0),
                'changes': prog.metadata.get('changes', ''),
            })
        
        return trace
    
    def _should_stop(self, metrics: ConvergenceMetrics) -> bool:
        """Check if we should stop the outer loop early"""
        
        # If early stopping is disabled, always run all iterations
        if self.meta_config.disable_early_stopping:
            return False
        
        # Patience exhausted
        if self.patience_counter >= self.meta_config.outer_patience:
            logger.info("Stopping: patience exhausted")
            return True
        
        # Target score reached
        if (self.meta_config.target_score is not None and
            self.best_score >= self.meta_config.target_score):
            logger.info(f"Stopping: target score {self.meta_config.target_score} reached")
            return True
        
        # Target convergence rate achieved
        if (self.meta_config.target_convergence_rate is not None and
            self.best_convergence_rate <= self.meta_config.target_convergence_rate):
            logger.info(f"Stopping: target convergence rate achieved")
            return True
        
        return False
    
    def _compare_to_baseline(self) -> Dict[str, float]:
        """Compare our convergence rate to literature baseline"""
        baseline_rate = (
            self.meta_config.literature_baseline_iterations /
            self.meta_config.literature_baseline_score
        )
        
        our_rate = self.best_convergence_rate
        if our_rate == float('inf'):
            our_rate = self.meta_config.inner_iterations_per_outer  # Use max iterations
        
        improvement = baseline_rate / our_rate if our_rate > 0 else 0
        
        return {
            'our_convergence_rate': our_rate,
            'baseline_convergence_rate': baseline_rate,
            'improvement_factor': improvement,
            'our_iterations_to_baseline_score': (
                our_rate * self.meta_config.literature_baseline_score
            ),
        }
    
    def _extract_problem_description(self) -> str:
        """Extract problem description from config or code"""
        # Try to get from existing system message
        existing = self.base_config.prompt.system_message
        if existing and len(existing) > 50:
            return existing
        
        # Fall back to generic description
        return "Optimize the given code to maximize the evaluation score while maintaining validity."
    
    def _extract_evaluation_criteria(self) -> str:
        """Extract evaluation criteria description"""
        return "Maximize 'combined_score' metric while ensuring 'validity' equals 1.0"
    
    def _extract_format_requirements(self) -> str:
        """
        Extract critical format requirements from the initial code.
        These requirements MUST be preserved by the generated prompts.
        """
        # Analyze initial code to find required function names
        required_functions = []
        for line in self.initial_code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('def ') and 'EVOLVE-BLOCK' not in line:
                # Function defined outside EVOLVE-BLOCK - must be preserved
                func_name = stripped[4:].split('(')[0]
                if func_name not in ['__init__', '__str__', '__repr__']:
                    required_functions.append(func_name)
        
        # Check for EVOLVE-BLOCK pattern
        has_evolve_block = 'EVOLVE-BLOCK-START' in self.initial_code
        
        # Find the main function that's called by external code
        # Usually it's the one that calls the evolved function
        main_function = None
        for line in self.initial_code.split('\n'):
            if 'def run_' in line:
                main_function = line.strip()[4:].split('(')[0]
                break
        
        # Build format requirements string
        requirements = []
        
        if has_evolve_block:
            requirements.append("- ONLY modify code within the EVOLVE-BLOCK markers")
            requirements.append("- DO NOT modify or remove code outside the EVOLVE-BLOCK")
        
        if main_function:
            requirements.append(f"- The function `{main_function}()` MUST exist and be callable")
            requirements.append(f"- DO NOT rename or remove the `{main_function}()` function")
        
        # Detect function calls in the main function to identify required evolved functions
        in_main = False
        for line in self.initial_code.split('\n'):
            if main_function and f'def {main_function}' in line:
                in_main = True
            elif in_main and line.strip().startswith('def '):
                in_main = False
            elif in_main and '(' in line:
                # Look for function calls
                for word in line.replace('(', ' ').replace(')', ' ').split():
                    if word.isidentifier() and word not in ['return', 'if', 'for', 'while', 'def']:
                        if any(f'def {word}' in self.initial_code for _ in [1]):
                            requirements.append(f"- The function `{word}()` MUST be defined (called by {main_function})")
                            break
        
        return "\n".join(requirements) if requirements else "Follow the existing code structure"
    
    def _save_prompt(self, prompt: str, outer_iter: int) -> None:
        """Save a seed prompt to file"""
        prompt_dir = os.path.join(self.output_dir, "prompts")
        os.makedirs(prompt_dir, exist_ok=True)
        
        path = os.path.join(prompt_dir, f"seed_prompt_{outer_iter}.txt")
        with open(path, 'w') as f:
            f.write(prompt)
    
    def _save_results(self, result: MetaEvolutionResult) -> None:
        """Save final results to file"""
        # Save summary
        summary_path = os.path.join(self.output_dir, "meta_evolution_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(result.summary())
        
        # Save best prompt
        if result.best_seed_prompt:
            best_prompt_path = os.path.join(self.output_dir, "best_seed_prompt.txt")
            with open(best_prompt_path, 'w') as f:
                f.write(result.best_seed_prompt)
        
        # Save metrics as JSON
        metrics_path = os.path.join(self.output_dir, "meta_evolution_metrics.json")
        metrics_data = {
            'best_score': result.best_score,
            'best_convergence_rate': result.best_convergence_rate,
            'total_inner_iterations': result.total_inner_iterations,
            'total_outer_iterations': result.total_outer_iterations,
            'total_wall_clock_time': result.total_wall_clock_time,
            'comparison_to_baseline': result.comparison_to_baseline,
            'iteration_scores': [r.best_score for r in result.iteration_history],
            'iteration_convergence_rates': [
                r.convergence_metrics.convergence_rate 
                for r in result.iteration_history
            ],
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved results to {self.output_dir}")

