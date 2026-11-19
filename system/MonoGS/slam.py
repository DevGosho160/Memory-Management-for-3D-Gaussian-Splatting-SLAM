import os
import json
import hashlib
import random
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import numpy as np
import torch.multiprocessing as mp
import yaml
from munch import munchify

try:
    import wandb
except ImportError:
    class DummyWandb:
        def init(self, *args, **kwargs): return self
        def log(self, *args, **kwargs): pass
        def Table(self, *args, **kwargs): return None
        def define_metric(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
    wandb = DummyWandb()
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.cpu_transfer_queue import CPUTransferQueue
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd

from project_utils.memory_limit import set_memory_limit


class SLAM:
    def __init__(self, config, save_dir=None, gpu_limit_mem=64.0, cpu_limit_mem=64.0):
        # Resolve limit from arguments or config (preserve budget even when pruning is disabled to allow OOM evaluation)
        if gpu_limit_mem == 64.0 or gpu_limit_mem is None:
            config_vram = config.get("vram_limit_gb") or config.get("MemoryManagement", {}).get("vram_limit_gb")
            if config_vram is not None:
                gpu_limit_mem = config_vram
        
        self.gpu_limit_mem = float(gpu_limit_mem) if gpu_limit_mem is not None else None
        self.cpu_limit_mem = float(cpu_limit_mem) if cpu_limit_mem is not None else None

        if self.gpu_limit_mem is not None or self.cpu_limit_mem is not None:
            set_memory_limit(gpu_limit_gb=self.gpu_limit_mem, cpu_limit_gb=self.cpu_limit_mem)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
        if config.get("Experiment", {}).get("cpu_transfer", False):
            frontend_queue = CPUTransferQueue(frontend_queue)
            backend_queue = CPUTransferQueue(backend_queue)

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.gpu_limit_mem = self.gpu_limit_mem
        self.frontend.set_hyperparams()

        # CPU-transfer mode constructs the initially empty model and background
        # in the child; sending either CUDA object at spawn would still use IPC.
        if config.get("Experiment", {}).get("cpu_transfer", False):
            self.backend.gaussians = None
            self.backend.background = None
        else:
            self.backend.gaussians = self.gaussians
            self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.gpu_limit_mem = self.gpu_limit_mem
        self.backend.save_dir = self.save_dir

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        self.frontend.backend_worker = backend_process
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        try:
            self.frontend.run()
        except BaseException:
            if backend_process.is_alive():
                backend_process.terminate()
            backend_process.join(timeout=5)
            for work_queue in (frontend_queue, backend_queue):
                work_queue.cancel_join_thread()
                work_queue.close()
            raise
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.save_dir is not None:
            with open(os.path.join(self.save_dir, "online_summary.json"), "w", encoding="utf-8") as file:
                json.dump({
                    "frames": N_frames,
                    "online_runtime_s": start.elapsed_time(end) * 0.001,
                    "online_fps": FPS,
                    "parent_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "parent_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "final_gaussian_count": self.frontend.gaussians.get_xyz.shape[0],
                    "retained_keyframes": len(self.frontend.kf_indices),
                }, file, indent=2)

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
                pose_ids=(list(range(len(self.dataset))) if
                          self.config.get("Evaluation", {}).get("pose_ids") == "all" else None),
                label_override=("fixed_final" if
                                self.config.get("Evaluation", {}).get("pose_ids") == "all" else None),
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
                rendering_ids=self.config.get("Evaluation", {}).get("rendering_ids"),
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            if self.config["Results"].get("color_refinement", True):
                # re-used the frontend queue to retrieve the gaussians from the backend.
                while not frontend_queue.empty():
                    frontend_queue.get()
                backend_queue.put(["color_refinement"])
                while True:
                    if frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = frontend_queue.get()
                    if data[0] == "sync_backend" and frontend_queue.empty():
                        self.gaussians = data[1]
                        break

                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    iteration="after_opt",
                )
                metrics_table.add_data(
                    "After",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
                )
            wandb.log({"Metrics": metrics_table})
            if self.config["Results"].get("color_refinement", True):
                save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        backend_queue.put(["stop"])
        # Drain frontend_queue to ensure child process feeder thread flushes cleanly and never deadlocks
        while not frontend_queue.empty():
            try:
                frontend_queue.get_nowait()
            except Exception:
                break

        backend_process.join(timeout=15)
        if backend_process.is_alive():
            backend_process.terminate()
            backend_process.join(timeout=3)
        Log("Backend stopped and joined the main thread")
        for q in (frontend_queue, backend_queue):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--online-eval", action="store_true",
                        help="Save online ATE and rendering without post-run refinement")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-memory-limits", action="store_true",
                        help="Disable the added per-process allocator/RLIMIT defaults")
    parser.add_argument("--cpu-transfer", action="store_true",
                        help="Opt-in CPU serialization for WSL CUDA IPC compatibility")
    parser.add_argument("--opacity-prune-threshold", type=float, default=None,
                        help="Opt-in extra opacity pruning after each keyframe map")
    parser.add_argument("--disable-memory-management", action="store_true",
                        help="Disable all memory management and pruning features to run unbounded into OOM")
    parser.add_argument("--memory-management", type=str, default=None,
                        choices=["none", "upstream", "custom"],
                        help="Memory management mode: 'none' (disabled/OOM), 'upstream', or 'custom' (bitwise controller)")
    parser.add_argument("--vram-limit-gb", type=float, default=None,
                        help="VRAM maximum budget in GB for the bitwise memory controller")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)

    if args.disable_memory_management or args.memory_management == "none":
        config["disable_memory_management"] = True
        config.setdefault("MemoryManagement", {})["disabled"] = True
        config["MemoryManagement"]["mode"] = "none"
        args.no_memory_limits = True
        Log(">>> MEMORY MANAGEMENT DISABLED: All pruning bypassed, system running unbounded towards OOM <<<")
    elif args.memory_management == "custom" or args.vram_limit_gb is not None:
        config.setdefault("MemoryManagement", {})["mode"] = "custom"
        config["use_custom_memory_controller"] = True
        if args.vram_limit_gb is not None:
            config["MemoryManagement"]["vram_limit_gb"] = args.vram_limit_gb
            config["vram_limit_gb"] = args.vram_limit_gb
            Log(f">>> CUSTOM BITWISE MEMORY CONTROLLER ACTIVE: VRAM limit = {args.vram_limit_gb} GB <<<")
    elif args.memory_management == "upstream":
        config.setdefault("MemoryManagement", {})["mode"] = "upstream"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config.setdefault("Experiment", {})["seed"] = args.seed
    config["Experiment"]["cpu_transfer"] = args.cpu_transfer
    if args.opacity_prune_threshold is not None:
        if not 0.0 < args.opacity_prune_threshold < 1.0:
            parser.error("--opacity-prune-threshold must be between 0 and 1")
        config["Experiment"]["opacity_prune_threshold"] = args.opacity_prune_threshold
    if config.get("Retention", {}).get("enabled", False):
        if config["Experiment"].get("opacity_prune_threshold") is not None:
            parser.error("Retention and extra opacity pruning cannot run together")
        if config["Retention"].get("policy") not in (
            "opacity", "random", "lru", "tracking_support"
        ):
            parser.error("Unknown Retention policy")
        if config["Retention"].get("max_gaussians", 0) < 1:
            parser.error("Retention.max_gaussians must be positive")
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if args.online_eval:
        config["Results"].update(save_results=True, use_gui=False,
                                 eval_rendering=True, use_wandb=False,
                                 color_refinement=False)

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        with open(os.path.join(save_dir, "run_manifest.json"), "w", encoding="utf-8") as file:
            json.dump({
                "argv": sys.argv,
                "cwd": os.getcwd(),
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
                    ["git", "diff", "HEAD"]
                )).hexdigest(),
                "git_status": subprocess.check_output(
                    ["git", "status", "--short"], text=True
                ).splitlines(),
                "seed": args.seed,
                "dataset_path": config["Dataset"]["dataset_path"],
                "resolved_config_sha256": hashlib.sha256(
                    yaml.safe_dump(config, sort_keys=True).encode()).hexdigest(),
                "retention": config.get("Retention"),
                "evaluation": config.get("Evaluation"),
                "status": "started",
            }, file, indent=2)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    try:
        limits = {"gpu_limit_mem": None, "cpu_limit_mem": None} if args.no_memory_limits else {}
        slam = SLAM(config, save_dir=save_dir, **limits)
        slam.run()
    except BaseException as exc:
        if save_dir is not None:
            with open(os.path.join(save_dir, "completion.json"), "w", encoding="utf-8") as file:
                json.dump({"status": "failed", "error": repr(exc)}, file, indent=2)
        raise
    else:
        if save_dir is not None:
            with open(os.path.join(save_dir, "completion.json"), "w", encoding="utf-8") as file:
                json.dump({"status": "completed"}, file, indent=2)
    finally:
        wandb.finish()

    # All done
    Log("Done.")
