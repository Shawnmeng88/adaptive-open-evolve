#!/usr/bin/env python3
"""
Experiment Runner: Compare Inner Loop vs Meta-Evolution

This script runs two experiments:
1. Inner loop only (standard OpenEvolve with fixed prompt)
2. Meta-evolution (outer loop that adapts prompts)

Then plots both convergence curves overlaid for comparison.

Usage:
    python run_experiment.py \
        --problem circle_packing \
        --model "meta-llama/Meta-Llama-3.1-405B-Instruct" \
        --total-iterations 200 \
        --outer-iterations 4 \
        --config configs/island_config_405b.yaml

This will run:
    - Inner loop: 200 iterations with fixed prompt
    - Meta-evolution: 4 outer × 50 inner = 200 total iterations
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add openevolve to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openevolve import OpenEvolve
from openevolve.config import Config, load_config

# Check if meta module exists
try:
    from openevolve.meta import MetaEvolutionController, MetaEvolutionConfig
    HAS_META = True
except ImportError:
    HAS_META = False
    print("Warning: Meta-evolution module not found. Only inner loop will be available.")


# Problem configurations
PROBLEMS = {
    "circle_packing": {
        "initial_program": "examples/circle_packing/initial_program.py",
        "evaluator": "examples/circle_packing/evaluator.py",
        "description": "Circle Packing (n=26)",
        "baseline_score": 2.635,
        "baseline_iterations": 1000,
    },
    "function_minimization": {
        "initial_program": "examples/function_minimization/initial_program.py",
        "evaluator": "examples/function_minimization/evaluator.py",
        "description": "Function Minimization",
        "baseline_score": None,
        "baseline_iterations": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comparison experiment: Inner Loop vs Meta-Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Problem selection
    parser.add_argument(
        "--problem", "-p",
        choices=list(PROBLEMS.keys()),
        default="circle_packing",
        help="Problem to solve (default: circle_packing)"
    )
    
    # Or custom paths
    parser.add_argument("--initial-program", help="Custom initial program path")
    parser.add_argument("--evaluator", help="Custom evaluator path")
    
    # Model and config
    parser.add_argument(
        "--model", "-m",
        default="meta-llama/Meta-Llama-3.1-405B-Instruct",
        help="LLM model to use"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/island_config_405b.yaml",
        help="Base config file"
    )
    
    # Iteration settings
    parser.add_argument(
        "--total-iterations", "-T",
        type=int,
        default=200,
        help="Total iterations for inner-loop-only run (default: 200)"
    )
    parser.add_argument(
        "--outer-iterations", "-O",
        type=int,
        default=4,
        help="Outer iterations for meta-evolution (default: 4)"
    )
    
    # What to run
    parser.add_argument(
        "--skip-inner", 
        action="store_true",
        help="Skip inner-loop-only experiment"
    )
    parser.add_argument(
        "--skip-meta",
        action="store_true", 
        help="Skip meta-evolution experiment"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        default="experiment_results",
        help="Output directory (default: experiment_results)"
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name (default: auto-generated)"
    )
    
    # Misc
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    
    return parser.parse_args()


def setup_logging(level: str, log_file: str):
    """Configure logging to both file and console"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ]
    )


def extract_convergence_trace(database) -> List[Dict]:
    """Extract iteration-by-iteration convergence data from database"""
    trace = []
    
    programs_by_iter = sorted(
        database.programs.values(),
        key=lambda p: p.iteration_found
    )
    
    best_score_so_far = 0.0
    
    for prog in programs_by_iter:
        score = prog.metrics.get('combined_score', 0.0)
        
        if score > best_score_so_far:
            best_score_so_far = score
        
        trace.append({
            'iteration': prog.iteration_found,
            'score': score,
            'best_so_far': best_score_so_far,
            'validity': prog.metrics.get('validity', 1.0),
        })
    
    return trace


def trace_to_best_curve(trace: List[Dict], total_iterations: int) -> Tuple[List[int], List[float]]:
    """Convert trace to best-so-far curve with regular iteration spacing"""
    if not trace:
        return list(range(total_iterations)), [0.0] * total_iterations
    
    # Create iteration -> best_so_far mapping
    iter_to_best = {}
    for entry in trace:
        it = entry['iteration']
        if it not in iter_to_best or entry['best_so_far'] > iter_to_best[it]:
            iter_to_best[it] = entry['best_so_far']
    
    # Fill in gaps with previous best
    iterations = list(range(total_iterations + 1))
    best_values = []
    current_best = 0.0
    
    for i in iterations:
        if i in iter_to_best:
            current_best = max(current_best, iter_to_best[i])
        best_values.append(current_best)
    
    return iterations, best_values


async def run_inner_loop_experiment(
    config: Config,
    initial_program: str,
    evaluator: str,
    total_iterations: int,
    output_dir: str,
) -> Tuple[List[int], List[float], float]:
    """
    Run inner-loop-only experiment (standard OpenEvolve)
    
    Returns:
        (iterations, best_scores, final_score)
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RUNNING INNER LOOP ONLY (Standard OpenEvolve)")
    logger.info(f"Total iterations: {total_iterations}")
    logger.info("=" * 80)
    
    inner_output = os.path.join(output_dir, "inner_loop_only")
    
    controller = OpenEvolve(
        initial_program_path=initial_program,
        evaluation_file=evaluator,
        config=config,
        output_dir=inner_output,
    )
    
    best_program = await controller.run(iterations=total_iterations)
    
    # Extract convergence trace
    trace = extract_convergence_trace(controller.database)
    iterations, best_scores = trace_to_best_curve(trace, total_iterations)
    
    final_score = best_program.metrics.get('combined_score', 0.0)
    
    # Save trace
    trace_path = os.path.join(inner_output, "convergence_trace.json")
    with open(trace_path, 'w') as f:
        json.dump({
            'iterations': iterations,
            'best_scores': best_scores,
            'final_score': final_score,
            'trace': trace,
        }, f, indent=2)
    
    logger.info(f"Inner loop complete. Final score: {final_score:.4f}")
    
    return iterations, best_scores, final_score


async def run_meta_evolution_experiment(
    config: Config,
    initial_program: str,
    evaluator: str,
    outer_iterations: int,
    inner_iterations_per_outer: int,
    output_dir: str,
) -> Tuple[List[int], List[float], float]:
    """
    Run meta-evolution experiment (outer loop + inner loops)
    
    Returns:
        (iterations, best_scores, final_score)
    """
    if not HAS_META:
        raise ImportError("Meta-evolution module not available")
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RUNNING META-EVOLUTION (Outer Loop + Inner Loops)")
    logger.info(f"Outer iterations: {outer_iterations}")
    logger.info(f"Inner iterations per outer: {inner_iterations_per_outer}")
    logger.info(f"Total iterations: {outer_iterations * inner_iterations_per_outer}")
    logger.info("=" * 80)
    
    meta_output = os.path.join(output_dir, "meta_evolution")
    
    meta_config = MetaEvolutionConfig(
        max_outer_iterations=outer_iterations,
        inner_iterations_per_outer=inner_iterations_per_outer,
        disable_early_stopping=True,  # Run all outer iterations
        meta_llm_model=config.llm.primary_model or config.llm.models[0].name,
        meta_llm_api_base=config.llm.api_base,
    )
    
    controller = MetaEvolutionController(
        base_config=config,
        initial_program_path=initial_program,
        evaluation_file=evaluator,
        meta_config=meta_config,
        output_dir=meta_output,
    )
    
    result = await controller.run()
    
    # Aggregate convergence across all inner runs
    all_iterations = []
    all_best_scores = []
    cumulative_iteration = 0
    running_best = 0.0
    
    for outer_result in result.iteration_history:
        # Load inner run trace
        inner_trace_path = os.path.join(
            meta_output, 
            f"inner_run_{outer_result.outer_iteration}",
            "convergence_trace.json"
        )
        
        if os.path.exists(inner_trace_path):
            with open(inner_trace_path, 'r') as f:
                inner_data = json.load(f)
            
            # Handle both dict format and list format
            best_scores = []
            if isinstance(inner_data, dict):
                best_scores = inner_data.get('best_scores', [])
            elif isinstance(inner_data, list):
                # List of trace entries - extract best_so_far values
                best_scores = [entry.get('best_so_far', 0.0) if isinstance(entry, dict) else 0.0 
                              for entry in inner_data]
            
            for i, score in enumerate(best_scores):
                running_best = max(running_best, score)
                all_iterations.append(cumulative_iteration + i)
                all_best_scores.append(running_best)
        
        cumulative_iteration += inner_iterations_per_outer
    
    final_score = result.best_score
    
    # Save aggregated trace
    trace_path = os.path.join(meta_output, "aggregated_convergence.json")
    with open(trace_path, 'w') as f:
        json.dump({
            'iterations': all_iterations,
            'best_scores': all_best_scores,
            'final_score': final_score,
            'outer_iterations': outer_iterations,
            'inner_per_outer': inner_iterations_per_outer,
        }, f, indent=2)
    
    logger.info(f"Meta-evolution complete. Final score: {final_score:.4f}")
    
    return all_iterations, all_best_scores, final_score


def plot_comparison(
    inner_data: Optional[Tuple[List[int], List[float], float]],
    meta_data: Optional[Tuple[List[int], List[float], float]],
    problem_name: str,
    model_name: str,
    output_path: str,
    baseline_score: Optional[float] = None,
):
    """Create comparison plot with overlaid curves"""
    
    plt.figure(figsize=(12, 7))
    
    # Style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'inner': '#2E86AB',  # Blue
        'meta': '#A23B72',   # Magenta
        'baseline': '#F18F01',  # Orange
    }
    
    # Plot inner loop
    if inner_data:
        iterations, scores, final = inner_data
        plt.plot(iterations, scores, 
                label=f'Inner Loop Only (final: {final:.4f})',
                color=colors['inner'], linewidth=2)
    
    # Plot meta-evolution
    if meta_data:
        iterations, scores, final = meta_data
        plt.plot(iterations, scores,
                label=f'Meta-Evolution (final: {final:.4f})',
                color=colors['meta'], linewidth=2, linestyle='--')
    
    # Plot baseline
    if baseline_score:
        plt.axhline(y=baseline_score, color=colors['baseline'], 
                   linestyle=':', linewidth=1.5,
                   label=f'Literature Baseline ({baseline_score:.3f})')
    
    # Labels and title
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Score', fontsize=12)
    
    # Shorten model name for title
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    plt.title(f'{problem_name}\nModel: {model_short}', fontsize=14, fontweight='bold')
    
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"\nPlot saved to: {output_path}")
    
    # Also show if in interactive mode
    try:
        plt.show()
    except:
        pass


def create_summary_report(
    inner_data: Optional[Tuple],
    meta_data: Optional[Tuple],
    problem_info: Dict,
    model_name: str,
    total_iterations: int,
    output_path: str,
):
    """Create a text summary report"""
    
    lines = [
        "=" * 70,
        "EXPERIMENT SUMMARY REPORT",
        "=" * 70,
        "",
        f"Problem: {problem_info.get('description', 'Unknown')}",
        f"Model: {model_name}",
        f"Total Iterations: {total_iterations}",
        f"Timestamp: {datetime.now().isoformat()}",
        "",
        "-" * 70,
        "RESULTS",
        "-" * 70,
        "",
    ]
    
    if inner_data:
        _, _, final_inner = inner_data
        lines.extend([
            "Inner Loop Only (Fixed Prompt):",
            f"  Final Score: {final_inner:.4f}",
        ])
    
    if meta_data:
        _, _, final_meta = meta_data
        lines.extend([
            "",
            "Meta-Evolution (Adaptive Prompts):",
            f"  Final Score: {final_meta:.4f}",
        ])
    
    if inner_data and meta_data:
        _, _, final_inner = inner_data
        _, _, final_meta = meta_data
        improvement = ((final_meta - final_inner) / final_inner * 100) if final_inner > 0 else 0
        lines.extend([
            "",
            "-" * 70,
            "COMPARISON",
            "-" * 70,
            f"Meta-Evolution improvement over Inner Loop: {improvement:+.2f}%",
        ])
        
        if final_meta > final_inner:
            lines.append("✓ Meta-Evolution achieved better results")
        elif final_meta < final_inner:
            lines.append("✗ Inner Loop achieved better results")
        else:
            lines.append("= Both methods achieved same results")
    
    if problem_info.get('baseline_score'):
        lines.extend([
            "",
            "-" * 70,
            "LITERATURE BASELINE",
            "-" * 70,
            f"Baseline Score: {problem_info['baseline_score']}",
            f"Baseline Iterations: {problem_info.get('baseline_iterations', 'N/A')}",
        ])
    
    lines.extend(["", "=" * 70])
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)


async def main():
    args = parse_args()
    
    # Determine problem paths
    if args.initial_program and args.evaluator:
        initial_program = args.initial_program
        evaluator = args.evaluator
        problem_info = {"description": "Custom Problem"}
    else:
        problem_info = PROBLEMS[args.problem]
        initial_program = problem_info["initial_program"]
        evaluator = problem_info["evaluator"]
    
    # Validate paths
    if not os.path.exists(initial_program):
        print(f"Error: Initial program not found: {initial_program}")
        sys.exit(1)
    if not os.path.exists(evaluator):
        print(f"Error: Evaluator not found: {evaluator}")
        sys.exit(1)
    
    # Create experiment directory
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split('/')[-1][:20]
        exp_name = f"{args.problem}_{model_short}_{args.total_iterations}iter_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, "experiment.log")
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    # Load config
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = Config()
    
    # Override model if specified
    if args.model:
        config.llm.primary_model = args.model
        config.llm.primary_model_weight = 1.0
        config.llm.secondary_model_weight = 0.0
        config.llm.rebuild_models()
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Problem: {problem_info.get('description', args.problem)}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Total iterations: {args.total_iterations}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    inner_data = None
    meta_data = None
    
    # Run inner loop experiment
    if not args.skip_inner:
        try:
            inner_data = await run_inner_loop_experiment(
                config=config,
                initial_program=initial_program,
                evaluator=evaluator,
                total_iterations=args.total_iterations,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.exception(f"Inner loop experiment failed: {e}")
    
    # Run meta-evolution experiment
    if not args.skip_meta and HAS_META:
        try:
            inner_per_outer = args.total_iterations // args.outer_iterations
            meta_data = await run_meta_evolution_experiment(
                config=config,
                initial_program=initial_program,
                evaluator=evaluator,
                outer_iterations=args.outer_iterations,
                inner_iterations_per_outer=inner_per_outer,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.exception(f"Meta-evolution experiment failed: {e}")
    
    # Create comparison plot
    if inner_data or meta_data:
        plot_path = os.path.join(output_dir, "convergence_comparison.png")
        plot_comparison(
            inner_data=inner_data,
            meta_data=meta_data,
            problem_name=problem_info.get('description', args.problem),
            model_name=args.model,
            output_path=plot_path,
            baseline_score=problem_info.get('baseline_score'),
        )
        
        # Create summary report
        report_path = os.path.join(output_dir, "summary_report.txt")
        create_summary_report(
            inner_data=inner_data,
            meta_data=meta_data,
            problem_info=problem_info,
            model_name=args.model,
            total_iterations=args.total_iterations,
            output_path=report_path,
        )
    
    print(f"\n✅ Experiment complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())

