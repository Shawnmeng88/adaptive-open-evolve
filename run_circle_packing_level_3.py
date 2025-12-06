import subprocess

CONFIG_FOLDER_PATH = "configs_circle_packing"

ISLANDS = [1, 3, 5]
OSS = [20, 120]
OUTER_ITERATIONS = [2, 5, 10]

# OSS = [120]
# OUTER_ITERATIONS = [2]


for islands in ISLANDS:
    for oss in OSS:
        for outer_iterations in OUTER_ITERATIONS:
            config_path = f"{CONFIG_FOLDER_PATH}/config_islands{islands}_oss{oss}.yaml"
            print(f"Running experiment with {islands} islands and {oss}b model")
            subprocess.run([
                "python3", 
                "run_experiment.py",
                "--problem", "circle_packing",
                "--model", "openai/gpt-oss-120b",
                "--analysis-model", "openai/gpt-oss-20b",
                "--config", config_path,
                "--total-iterations", "50",
                "--outer-iterations", f"{outer_iterations}",
                "--output-dir", f"circle_packing_full_experiment_level3/experiment_results_circle_packing_islands{islands}_oss{oss}_outer{outer_iterations}",
                "--skip-inner",
                "--verbose-prompts"
            ])
