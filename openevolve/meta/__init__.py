"""
Meta-Evolution Framework for Automatic Prompt Engineering

This module implements a double-nested loop framework where:
- Outer loop: A Meta-LLM generates and refines seed prompts based on convergence feedback
- Inner loop: Standard OpenEvolve evolves programs using the current seed prompt

Usage:
    from openevolve.meta import MetaEvolutionController, MetaEvolutionConfig
    
    meta_config = MetaEvolutionConfig(
        max_outer_iterations=10,
        inner_iterations_per_outer=100,
    )
    
    controller = MetaEvolutionController(
        base_config=config,
        initial_program_path="program.py",
        evaluation_file="evaluator.py",
        meta_config=meta_config,
    )
    
    result = await controller.run()
"""

from openevolve.meta.meta_controller import MetaEvolutionController, MetaEvolutionConfig, MetaEvolutionResult
from openevolve.meta.convergence import ConvergenceAnalyzer, ConvergenceMetrics
from openevolve.meta.meta_llm import MetaLLM
from openevolve.meta.seed_prompt import SeedPromptHistory
from openevolve.meta.code_analyzer import CodeAnalyzer, CodeAnalysisResult

__all__ = [
    "MetaEvolutionController",
    "MetaEvolutionConfig", 
    "MetaEvolutionResult",
    "ConvergenceAnalyzer",
    "ConvergenceMetrics",
    "MetaLLM",
    "SeedPromptHistory",
    "CodeAnalyzer",
    "CodeAnalysisResult",
]

