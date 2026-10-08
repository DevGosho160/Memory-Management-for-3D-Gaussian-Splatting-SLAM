import os
import sys
import csv
import json
import yaml
import time
import subprocess
from pathlib import Path

def create_runner_script():
    # We write a custom runner because slam.py does not expose gpu_limit_mem via argparse,
    # and we need a fresh process for each memory limit (PyTorch requirement).
    runner_code = """
import os
import sys
import yaml
import json
from argparse import ArgumentParser
import torch
import torch.multiprocessing as mp

# Import SLAM and config utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'MonoGS_System', 'MonoGS'))
from slam import SLAM
from utils.config_utils import load_config
from utils.eval_utils import eval_ate

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu-limit", type=str, default="None")
    parser.add_argument("--save-dir", type=str, required=True)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    config = load_config(args.config)
    config["Results"]["save_results"] = True
    config["Results"]["use_gui"] = False
    config["Results"]["eval_rendering"] = False
    config["Results"]["use_wandb"] = False
    config["Results"]["color_refinement"] = False
    
    # Optional: If you want to use the retention system to avoid OOMs instead of just hard-crashing:
    # config["Retention"]["enabled"] = True
    
    gpu_limit = float(args.gpu_limit) if args.gpu_limit != "None" else None

    # Run SLAM
    try:
        slam = SLAM(config, save_dir=args.save_dir, gpu_limit_mem=gpu_limit)
        slam.run()
        
        # SLAM run finishes. ATE is calculated in slam.py if eval_rendering is true, 
        # but we can just calculate it here to be safe if it completed successfully.
        ate = eval_ate(
            slam.frontend.cameras,
            slam.frontend.kf_indices,
            args.save_dir,
            0,
            final=True,
            monocular=config["Dataset"]["sensor_type"] == "monocular",
        )
        print(f"RESULTS_ATE:{ate}")
    except Exception as e:
        print(f"RESULTS_ERROR:{e}")
        sys.exit(1)
"""
    with open("temp_runner.py", "w") as f:
        f.write(runner_code)

def main():
    tum_configs = [
        "MonoGS_System/MonoGS/configs/rgbd/tum/fr1_desk.yaml",
        "MonoGS_System/MonoGS/configs/rgbd/tum/fr2_xyz.yaml",
        "MonoGS_System/MonoGS/configs/rgbd/tum/fr3_office.yaml"
    ]
    
    methods = {
        "Opacity": "Opacity",
        "Volume": "Volume",
        "Visibility": "Visibility",
        "VoxelGrid": "VoxelGrid",
        "Density": "Density"
    }
    
    vram_limits = [0.5, 1.0]
    
    output_csv = "tum_benchmark_results.csv"
    
    create_runner_script()
    
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Dataset", "Pruning Method", "VRAM Limit (GB)", "ATE", "Processing Time (s)", "Status"])
        
        for config_path in tum_configs:
            if not os.path.exists(config_path):
                print(f"Config {config_path} not found. Skipping.")
                continue
                
            dataset_name = Path(config_path).stem
            
            for method_name, policy in methods.items():
                for vram in vram_limits:
                    print(f"Running {dataset_name} | {method_name} | VRAM: {vram}GB")
                    
                    # Create a temporary config for this run
                    with open(config_path, "r") as f:
                        config_data = yaml.safe_load(f)
                    
                    # Ensure Retention is enabled and set policy
                    if "Retention" not in config_data:
                        config_data["Retention"] = {}
                    config_data["Retention"]["enabled"] = True
                    config_data["Retention"]["policy"] = policy
                    
                    if "Dataset" not in config_data:
                        config_data["Dataset"] = {}
                    if "Training" not in config_data:
                        config_data["Training"] = {}
                    config_data["Dataset"]["single_thread"] = True
                    config_data["Training"]["single_thread"] = True
                    
                    # Estimate max_gaussians to avoid OOM (roughly 300 bytes per gaussian, but overhead is larger)
                    # This is optional, but helps the retention policy actually do its job before PyTorch OOMs.
                    if vram != "None":
                        config_data["Retention"]["memory_limit_gb"] = vram
                    else:
                        config_data["Retention"]["enabled"] = False # Disable for unlimited
                    
                    temp_config_path = os.path.abspath(f"temp_config_{dataset_name}.yaml")
                    with open(temp_config_path, "w") as f:
                        yaml.dump(config_data, f)
                        
                    save_dir = os.path.abspath(f"benchmark_results/{dataset_name}_{method_name}_{vram}GB")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    start_time = time.time()
                    
                    # Execute runner from the MonoGS directory so relative config paths work
                    process = subprocess.Popen(
                        [sys.executable, os.path.abspath("temp_runner.py"), 
                         "--config", temp_config_path, 
                         "--gpu-limit", str(vram),
                         "--save-dir", save_dir],
                        cwd=os.path.abspath("MonoGS_System/MonoGS"),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    stdout, stderr = process.communicate()
                    processing_time = time.time() - start_time
                    
                    # Parse results
                    ate = "N/A"
                    status = "Success"
                    
                    if process.returncode != 0:
                        status = "Failed/Crashed"
                        print(f"Run failed. Stderr: {stderr[-500:]}")
                    else:
                        for line in stdout.splitlines():
                            if line.startswith("RESULTS_ATE:"):
                                ate = line.split("RESULTS_ATE:")[1].strip()
                            elif line.startswith("RESULTS_ERROR:"):
                                status = f"Error: {line.split('RESULTS_ERROR:')[1].strip()}"
                                
                    print(f"Result -> ATE: {ate}, Time: {processing_time:.2f}s, Status: {status}")
                    writer.writerow([dataset_name, method_name, str(vram), ate, f"{processing_time:.2f}", status])
                    csv_file.flush()
                    
                    # Cleanup temp config
                    if os.path.exists(temp_config_path):
                        os.remove(temp_config_path)

    if os.path.exists("temp_runner.py"):
        os.remove("temp_runner.py")
    
    print(f"Benchmark completed. Results saved to {output_csv}")

if __name__ == "__main__":
    main()
