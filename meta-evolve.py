#!/usr/bin/env python3
"""
Meta-Evolution Runner

Run the double-nested loop framework for automatic prompt engineering.

Usage:
    python meta-evolve.py <initial_program> <evaluator> [options]
    
Example:
    python meta-evolve.py \
        examples/circle_packing/initial_program.py \
        examples/circle_packing/evaluator.py \
        --config examples/circle_packing/meta_config.yaml \
        --outer-iterations 5 \
        --inner-iterations 50
"""

import argparse
import asyncio
import logging
import os
import sys

# Add openevolve to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openevolve.config import Config
from openevolve.meta import MetaEvolutionController, MetaEvolutionConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run meta-evolution for automatic prompt engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python meta-evolve.py program.py evaluator.py

  # Custom iterations
  python meta-evolve.py program.py evaluator.py --outer-iterations 5 --inner-iterations 50

  # With config file
  python meta-evolve.py program.py evaluator.py --config config.yaml

  # Compare to literature baseline
  python meta-evolve.py program.py evaluator.py \\
      --baseline-iterations 1000 --baseline-score 2.635
        """
    )
    
    # Required arguments
    parser.add_argument(
        "initial_program",
        help="Path to the initial program to evolve"
    )
    parser.add_argument(
        "evaluator",
        help="Path to the evaluator script"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        help="Path to OpenEvolve config YAML file"
    )
    parser.add_argument(
        "--meta-config",
        help="Path to meta-evolution config YAML file"
    )
    
    # Outer loop parameters
    parser.add_argument(
        "--outer-iterations", "-O",
        type=int,
        default=10,
        help="Maximum outer loop iterations (default: 10)"
    )
    parser.add_argument(
        "--inner-iterations", "-I",
        type=int,
        default=100,
        help="Inner loop iterations per outer iteration (default: 100)"
    )
    parser.add_argument(
        "--outer-patience",
        type=int,
        default=3,
        help="Outer iterations without improvement before stopping (default: 3)"
    )
    
    # Target stopping conditions
    parser.add_argument(
        "--target-score",
        type=float,
        help="Stop if this score is achieved"
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        help="Stop if this convergence rate is achieved"
    )
    
    # Literature baseline comparison
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        help="Literature baseline: iterations to convergence"
    )
    parser.add_argument(
        "--baseline-score",
        type=float,
        help="Literature baseline: final score achieved"
    )
    
    # Meta-LLM settings
    parser.add_argument(
        "--meta-model",
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="Model to use for meta-LLM (default: Llama 70B)"
    )
    parser.add_argument(
        "--meta-temperature",
        type=float,
        default=0.7,
        help="Temperature for meta-LLM (default: 0.7)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", "-o",
        default="meta_evolution_output",
        help="Output directory (default: meta_evolution_output)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()


def setup_logging(level: str):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


async def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate input files exist
    if not os.path.exists(args.initial_program):
        logger.error(f"Initial program not found: {args.initial_program}")
        sys.exit(1)
    
    if not os.path.exists(args.evaluator):
        logger.error(f"Evaluator not found: {args.evaluator}")
        sys.exit(1)
    
    # Load base OpenEvolve config
    if args.config and os.path.exists(args.config):
        base_config = Config.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        base_config = Config()
        logger.info("Using default OpenEvolve config")
    
    # Create meta-evolution config
    meta_config = MetaEvolutionConfig(
        max_outer_iterations=args.outer_iterations,
        inner_iterations_per_outer=args.inner_iterations,
        outer_patience=args.outer_patience,
        target_score=args.target_score,
        target_convergence_rate=args.target_rate,
        meta_llm_model=args.meta_model,
        meta_llm_temperature=args.meta_temperature,
        literature_baseline_iterations=args.baseline_iterations,
        literature_baseline_score=args.baseline_score,
    )
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("META-EVOLUTION FRAMEWORK")
    logger.info("Automatic Prompt Engineering and Adaptation")
    logger.info("=" * 80)
    logger.info(f"Initial program: {args.initial_program}")
    logger.info(f"Evaluator: {args.evaluator}")
    logger.info(f"Outer iterations: {meta_config.max_outer_iterations}")
    logger.info(f"Inner iterations per outer: {meta_config.inner_iterations_per_outer}")
    logger.info(f"Meta-LLM model: {meta_config.meta_llm_model}")
    logger.info(f"Output directory: {output_dir}")
    
    if args.baseline_iterations and args.baseline_score:
        logger.info(f"Literature baseline: {args.baseline_iterations} iterations to score {args.baseline_score}")
    
    logger.info("=" * 80)
    
    # Create and run meta-evolution controller
    controller = MetaEvolutionController(
        base_config=base_config,
        initial_program_path=args.initial_program,
        evaluation_file=args.evaluator,
        meta_config=meta_config,
        output_dir=output_dir,
    )
    
    try:
        result = await controller.run()
        
        # Print final summary
        print("\n" + result.summary())
        
        # Print key files
        print(f"\nKey output files:")
        print(f"  Best seed prompt: {output_dir}/best_seed_prompt.txt")
        print(f"  Metrics: {output_dir}/meta_evolution_metrics.json")
        print(f"  Prompt history: {output_dir}/prompt_history.json")
        print(f"  Summary: {output_dir}/meta_evolution_summary.txt")
        
        if result.comparison_to_baseline:
            improvement = result.comparison_to_baseline['improvement_factor']
            if improvement > 1:
                print(f"\n🎉 Achieved {improvement:.2f}x faster convergence than baseline!")
            else:
                print(f"\n📊 Convergence was {1/improvement:.2f}x slower than baseline")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Meta-evolution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

