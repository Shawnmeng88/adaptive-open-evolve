#!/bin/bash
# Run circle packing example with Sophia endpoints

# Change to the project root
cd "$(dirname "$0")/../.."

# Get Sophia access token and set as OPENAI_API_KEY
# OpenEvolve uses OPENAI_API_KEY for authentication
export OPENAI_API_KEY=$(python /Users/shawnmeng/Desktop/Projects/CMSC35200/inference_auth_token.py get_access_token)

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: Could not get Sophia access token"
    echo "Please run: python /Users/shawnmeng/Desktop/Projects/CMSC35200/inference_auth_token.py authenticate"
    exit 1
fi

echo "Got Sophia access token (expires in $(python /Users/shawnmeng/Desktop/Projects/CMSC35200/inference_auth_token.py get_time_until_token_expiration --units minutes) minutes)"

# Run OpenEvolve with circle packing
python openevolve-run.py \
    examples/circle_packing/initial_program.py \
    examples/circle_packing/evaluator.py \
    --config examples/circle_packing/config_sophia.yaml \
    --iterations ${1:-50}

echo "Done! Check examples/circle_packing/openevolve_output/ for results"

