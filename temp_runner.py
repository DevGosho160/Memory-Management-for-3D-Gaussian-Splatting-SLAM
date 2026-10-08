
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
