#!/bin/bash
# Run meta-evolution for circle packing problem
#
# This demonstrates the automatic prompt engineering framework.
# The outer loop (Meta-LLM) learns better seed prompts based on 
# convergence feedback from the inner loop (OpenEvolve).
#
# Literature baseline from AlphaEvolve paper:
#   - Target score: 2.635 (sum of radii for n=26)
#   - Typical convergence: ~1000 iterations

set -e

# Navigate to openevolve root
cd "$(dirname "$0")/../.."

# Get API token from Sophia (if using ALCF endpoint)
# Uncomment and modify if needed:
# export OPENAI_API_KEY="$(python3 configs/inference_auth_token.py get_access_token)"

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. Please export your API key."
    echo "  For Sophia: export OPENAI_API_KEY=\"\$(python3 configs/inference_auth_token.py get_access_token)\""
    echo "  For OpenAI: export OPENAI_API_KEY=\"your-key\""
    exit 1
fi

# Default parameters (can be overridden via command line)
OUTER_ITERATIONS=${1:-5}
INNER_ITERATIONS=${2:-50}

echo "============================================================"
echo "META-EVOLUTION: Automatic Prompt Engineering"
echo "============================================================"
echo "Problem: Circle Packing (n=26)"
echo "Outer iterations: $OUTER_ITERATIONS"
echo "Inner iterations per outer: $INNER_ITERATIONS"
echo "Total iterations: $((OUTER_ITERATIONS * INNER_ITERATIONS))"
echo ""
echo "Literature baseline: 1000 iterations for score 2.635"
echo "============================================================"
echo ""

# Run meta-evolution
python3 meta-evolve.py \
    examples/circle_packing/initial_program.py \
    examples/circle_packing/evaluator.py \
    --config examples/circle_packing/meta_config.yaml \
    --outer-iterations "$OUTER_ITERATIONS" \
    --inner-iterations "$INNER_ITERATIONS" \
    --outer-patience 3 \
    --baseline-iterations 1000 \
    --baseline-score 2.635 \
    --output-dir examples/circle_packing/meta_evolution_output \
    --log-level INFO

echo ""
echo "============================================================"
echo "Meta-evolution complete!"
echo "Results saved to: examples/circle_packing/meta_evolution_output/"
echo "============================================================"

