#!/usr/bin/env python3
"""
Multi-Run Experiment Runner: Run multiple experiments and average results

This script runs circle packing experiments multiple times and creates
averaged convergence plots for both inner loop and meta-evolution.

Usage:
    python run_multi_experiment.py \
        --num-runs 5 \
        --problem circle_packing \
        --model "openai/gpt-oss-120b" \
        --total-iterations 200 \
        --outer-iterations 4 \
        --config configs/island_config_405b.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
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
    print("Warning: Meta-evolution module not available.")


# Problem configurations
PROBLEMS = {
    "circle_packing": {
        "initial_program": "examples/circle_packing/initial_program.py",
        "evaluator": "examples/circle_packing/evaluator.py",
        "description": "Circle Packing (n=26)",
        "baseline_score": 1.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments and average results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--num-runs", "-n",
        type=int,
        default=5,
        help="Number of experimental runs to average (default: 5)"
    )
    
    parser.add_argument(
        "--problem", "-p",
        choices=list(PROBLEMS.keys()),
        default="circle_packing",
        help="Problem to solve (default: circle_packing)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="openai/gpt-oss-120b",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="configs/island_config_405b.yaml",
        help="Base config file"
    )
    
    parser.add_argument(
        "--total-iterations", "-T",
        type=int,
        default=200,
        help="Total iterations for inner-loop-only run"
    )
    
    parser.add_argument(
        "--outer-iterations", "-O",
        type=int,
        default=4,
        help="Outer iterations for meta-evolution"
    )
    
    parser.add_argument(
        "--skip-inner",
        action="store_true",
        help="Skip inner-loop-only experiments"
    )
    
    parser.add_argument(
        "--skip-meta",
        action="store_true",
        help="Skip meta-evolution experiments"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="multi_run_results",
        help="Output directory (default: multi_run_results)"
    )
    
    parser.add_argument(
        "--analysis-model",
        type=str,
        default=None,
        help="LLM model for code analysis"
    )
    
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
        return list(range(total_iterations + 1)), [0.0] * (total_iterations + 1)
    
    iter_to_best = {}
    for entry in trace:
        it = entry['iteration']
        if it not in iter_to_best or entry['best_so_far'] > iter_to_best[it]:
            iter_to_best[it] = entry['best_so_far']
    
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
    run_num: int,
) -> Tuple[List[int], List[float], float]:
    """Run inner-loop-only experiment"""
    import copy
    
    logger = logging.getLogger(__name__)
    logger.info(f"Run {run_num}: Inner loop only")
    
    inner_output = os.path.join(output_dir, f"run_{run_num}", "inner_loop_only")
    os.makedirs(inner_output, exist_ok=True)
    
    inner_config = copy.deepcopy(config)
    
    controller = OpenEvolve(
        initial_program_path=initial_program,
        evaluation_file=evaluator,
        config=inner_config,
        output_dir=inner_output,
    )
    
    best_program = await controller.run(iterations=total_iterations)
    
    trace = extract_convergence_trace(controller.database)
    iterations, best_scores = trace_to_best_curve(trace, total_iterations)
    
    final_score = 0.0
    if best_program and best_program.metrics:
        final_score = best_program.metrics.get('combined_score', 0.0)
    
    return iterations, best_scores, final_score


async def run_meta_evolution_experiment(
    config: Config,
    initial_program: str,
    evaluator: str,
    outer_iterations: int,
    inner_iterations_per_outer: int,
    output_dir: str,
    run_num: int,
    analysis_model: Optional[str] = None,
) -> Tuple[List[int], List[float], float]:
    """Run meta-evolution experiment"""
    if not HAS_META:
        raise ImportError("Meta-evolution module not available")
    
    logger = logging.getLogger(__name__)
    total_iterations = outer_iterations * inner_iterations_per_outer
    
    logger.info(f"Run {run_num}: Meta-evolution")
    
    meta_output = os.path.join(output_dir, f"run_{run_num}", "meta_evolution")
    os.makedirs(meta_output, exist_ok=True)
    
    import copy
    meta_base_config = copy.deepcopy(config)
    
    meta_config = MetaEvolutionConfig(
        max_outer_iterations=outer_iterations,
        inner_iterations_per_outer=inner_iterations_per_outer,
        disable_early_stopping=True,
        meta_llm_model=meta_base_config.llm.primary_model or meta_base_config.llm.models[0].name,
        meta_llm_api_base=meta_base_config.llm.api_base,
        save_convergence_traces=True,
        save_all_prompts=False,  # Skip verbose prompts for multi-run
        verbose_prompts=False,
        code_analysis_model=analysis_model,
    )
    
    meta_controller = MetaEvolutionController(
        base_config=meta_base_config,
        initial_program_path=initial_program,
        evaluation_file=evaluator,
        meta_config=meta_config,
        output_dir=meta_output,
    )
    
    result = await meta_controller.run()
    
    full_trace = extract_convergence_trace(meta_controller.controller.database)
    iterations, best_scores = trace_to_best_curve(full_trace, total_iterations)
    
    final_score = result.best_score
    
    return iterations, best_scores, final_score


def average_curves(all_curves: List[Tuple[List[int], List[float], float]], max_iterations: int) -> Tuple[List[int], List[float], float, List[float]]:
    """
    Average multiple convergence curves
    
    Returns:
        (iterations, mean_scores, mean_final_score, std_scores)
    """
    if not all_curves:
        return [], [], 0.0, []
    
    # Align all curves to the same iteration grid
    iterations = list(range(max_iterations + 1))
    
    # Pad or truncate each curve to max_iterations
    aligned_scores = []
    final_scores = []
    
    for iter_list, scores, final_score in all_curves:
        # Extend or pad to max_iterations
        if len(scores) < len(iterations):
            # Pad with last value
            padded = scores + [scores[-1]] * (len(iterations) - len(scores))
        else:
            # Truncate
            padded = scores[:len(iterations)]
        
        aligned_scores.append(padded)
        final_scores.append(final_score)
    
    # Convert to numpy arrays for easy averaging
    aligned_scores = np.array(aligned_scores)
    
    mean_scores = np.mean(aligned_scores, axis=0).tolist()
    std_scores = np.std(aligned_scores, axis=0).tolist()
    mean_final = np.mean(final_scores)
    
    return iterations, mean_scores, mean_final, std_scores


def plot_averaged_comparison(
    inner_data: Optional[Tuple[List[int], List[float], float, List[float]]],
    meta_data: Optional[Tuple[List[int], List[float], float, List[float]]],
    problem_name: str,
    model_name: str,
    output_path: str,
    num_runs: int,
    baseline_score: Optional[float] = None,
):
    """Create averaged comparison plot with no tick marks"""
    
    plt.figure(figsize=(12, 7))
    
    # Style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'inner': '#2E86AB',  # Blue
        'meta': '#A23B72',   # Magenta
        'baseline': '#F18F01',  # Orange
    }
    
    # Plot inner loop (mean + std)
    if inner_data:
        iterations, mean_scores, mean_final, std_scores = inner_data
        mean_scores = np.array(mean_scores)
        std_scores = np.array(std_scores)
        
        plt.plot(iterations, mean_scores, 
                label=f'Inner Loop Only (mean final: {mean_final:.4f}, n={num_runs})',
                color=colors['inner'], linewidth=2.5)
        
        # Add shaded std region
        plt.fill_between(iterations, 
                        mean_scores - std_scores, 
                        mean_scores + std_scores,
                        color=colors['inner'], alpha=0.2)
    
    # Plot meta-evolution (mean + std)
    if meta_data:
        iterations, mean_scores, mean_final, std_scores = meta_data
        mean_scores = np.array(mean_scores)
        std_scores = np.array(std_scores)
        
        plt.plot(iterations, mean_scores,
                label=f'Meta-Evolution (mean final: {mean_final:.4f}, n={num_runs})',
                color=colors['meta'], linewidth=2.5, linestyle='--')
        
        # Add shaded std region
        plt.fill_between(iterations,
                        mean_scores - std_scores,
                        mean_scores + std_scores,
                        color=colors['meta'], alpha=0.2)
    
    # Plot baseline
    if baseline_score:
        plt.axhline(y=baseline_score, color=colors['baseline'], 
                   linestyle=':', linewidth=1.5,
                   label=f'Literature Baseline ({baseline_score:.3f})')
    
    # Labels and title
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Score', fontsize=12)
    
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    plt.title(f'{problem_name}\nModel: {model_short} (averaged over {num_runs} runs)', 
              fontsize=14, fontweight='bold')
    
    plt.legend(loc='lower right', fontsize=10)
    
    # Remove tick marks
    ax = plt.gca()
    ax.tick_params(left=False, bottom=False, top=False, right=False)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"\nPlot saved to: {output_path}")


async def main():
    args = parse_args()
    
    # Determine problem paths
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
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split('/')[-1][:20]
    exp_name = f"{args.problem}_{model_short}_{args.num_runs}runs_{timestamp}"
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
    logger.info(f"MULTI-RUN EXPERIMENT: {args.num_runs} runs")
    logger.info("=" * 80)
    logger.info(f"Problem: {problem_info['description']}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Total iterations: {args.total_iterations}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    # Collect all runs
    all_inner_data = []
    all_meta_data = []
    
    inner_per_outer = args.total_iterations // args.outer_iterations if args.outer_iterations > 0 else 0
    
    # Run all experiments
    for run_num in range(1, args.num_runs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"RUN {run_num}/{args.num_runs}")
        logger.info(f"{'='*80}")
        
        # Inner loop experiments
        if not args.skip_inner:
            try:
                inner_data = await run_inner_loop_experiment(
                    config=config,
                    initial_program=initial_program,
                    evaluator=evaluator,
                    total_iterations=args.total_iterations,
                    output_dir=output_dir,
                    run_num=run_num,
                )
                all_inner_data.append(inner_data)
                logger.info(f"Run {run_num} inner loop: final={inner_data[2]:.4f}")
            except Exception as e:
                logger.exception(f"Run {run_num} inner loop failed: {e}")
        
        # Meta-evolution experiments
        if not args.skip_meta and HAS_META:
            try:
                meta_data = await run_meta_evolution_experiment(
                    config=config,
                    initial_program=initial_program,
                    evaluator=evaluator,
                    outer_iterations=args.outer_iterations,
                    inner_iterations_per_outer=inner_per_outer,
                    output_dir=output_dir,
                    run_num=run_num,
                    analysis_model=args.analysis_model,
                )
                all_meta_data.append(meta_data)
                logger.info(f"Run {run_num} meta-evolution: final={meta_data[2]:.4f}")
            except Exception as e:
                logger.exception(f"Run {run_num} meta-evolution failed: {e}")
    
    # Average the curves
    logger.info("\n" + "=" * 80)
    logger.info("AVERAGING RESULTS")
    logger.info("=" * 80)
    
    averaged_inner = None
    averaged_meta = None
    
    if all_inner_data:
        max_iter = max(len(iter_list) - 1 for iter_list, _, _ in all_inner_data)
        averaged_inner = average_curves(all_inner_data, max_iter)
        logger.info(f"Averaged {len(all_inner_data)} inner loop runs")
    
    if all_meta_data:
        max_iter = max(len(iter_list) - 1 for iter_list, _, _ in all_meta_data)
        averaged_meta = average_curves(all_meta_data, max_iter)
        logger.info(f"Averaged {len(all_meta_data)} meta-evolution runs")
    
    # Create averaged plot
    if averaged_inner or averaged_meta:
        plot_path = os.path.join(output_dir, "averaged_convergence.png")
        
        plot_averaged_comparison(
            inner_data=averaged_inner,
            meta_data=averaged_meta,
            problem_name=problem_info.get('description', args.problem),
            model_name=args.model,
            output_path=plot_path,
            num_runs=args.num_runs,
            baseline_score=problem_info.get('baseline_score'),
        )
        
        # Save summary
        summary = {
            'num_runs': args.num_runs,
            'problem': args.problem,
            'model': args.model,
            'total_iterations': args.total_iterations,
            'outer_iterations': args.outer_iterations,
            'inner_per_outer': inner_per_outer,
        }
        
        if averaged_inner:
            _, _, mean_final, _ = averaged_inner
            summary['inner_loop'] = {
                'num_runs': len(all_inner_data),
                'mean_final_score': mean_final,
            }
        
        if averaged_meta:
            _, _, mean_final, _ = averaged_meta
            summary['meta_evolution'] = {
                'num_runs': len(all_meta_data),
                'mean_final_score': mean_final,
            }
        
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✅ Multi-run experiment complete!")
        logger.info(f"Results saved to: {output_dir}")
    else:
        logger.error("No data collected! Cannot create plot.")


if __name__ == "__main__":
    asyncio.run(main())