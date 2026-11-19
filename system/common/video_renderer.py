"""Video Renderer for 3D Gaussian Splatting SLAM Reconstructions.

Renders the reconstructed 3D Gaussian Splatting map moving through the given environment/dataset
along the camera trajectory (or smooth orbiting camera path) and saves the result as MP4 / GIF.
"""

import os
from typing import List, Optional, Any, Union
import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio
except ImportError:
    imageio = None


class MapVideoRenderer:
    """Renders reconstructed map views and compiles them into a trajectory video."""

    def __init__(self, output_dir: str, fps: int = 30):
        self.output_dir = output_dir
        self.fps = int(fps)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def smooth_frames(frames: List[np.ndarray], alpha: float = 0.20) -> List[np.ndarray]:
        """Applies gentle temporal blending to eliminate high-frequency rendering flicker."""
        if len(frames) <= 2:
            return frames
        smoothed = [frames[0]]
        prev = frames[0].astype(np.float32)
        for f in frames[1:]:
            curr = f.astype(np.float32)
            blended = (1.0 - alpha) * curr + alpha * prev
            smoothed.append(np.clip(blended, 0, 255).astype(np.uint8))
            prev = blended
        return smoothed

    def render_trajectory_video(
        self,
        rendered_images: Optional[List[np.ndarray]] = None,
        model: Optional[Any] = None,
        cameras: Optional[List[Any]] = None,
        video_filename: str = "reconstructed_map_trajectory.mp4",
        fallback_gif: str = "reconstructed_map_trajectory.gif",
        apply_smoothing: bool = True,
    ) -> Optional[str]:
        """
        Takes either a list of pre-rendered image arrays (uint8 HxWxC) or renders them from
        the model + cameras, and writes out a video file.
        """
        frames: List[np.ndarray] = []

        if rendered_images is not None and len(rendered_images) > 0:
            for img in rendered_images:
                if isinstance(img, torch.Tensor):
                    arr = img.detach().cpu().numpy()
                    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
                        arr = np.transpose(arr, (1, 2, 0))
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    frames.append(arr)
                elif isinstance(img, np.ndarray):
                    arr = img
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    frames.append(arr)

        elif model is not None and cameras is not None:
            # Render on the fly using model and cameras
            try:
                from gaussian_splatting.gaussian_renderer import render
                from gaussian_splatting.utils.system_utils import searchForMaxIteration
                
                pipeline = None
                bg = torch.zeros(3, device="cuda" if torch.cuda.is_available() else "cpu")
                for cam in cameras:
                    with torch.no_grad():
                        pkg = render(cam, model, pipeline, bg)
                        rendered = pkg["render"].detach().cpu().permute(1, 2, 0).numpy()
                        arr = np.clip(rendered * 255.0, 0, 255).astype(np.uint8)
                        frames.append(arr)
            except Exception as e:
                print(f"[MapVideoRenderer] Direct render failed: {e}")

        if not frames:
            print("[MapVideoRenderer] No frames available to render video.")
            return None

        if apply_smoothing and len(frames) > 2:
            frames = self.smooth_frames(frames)

        video_path = os.path.join(self.output_dir, video_filename)
        H, W = frames[0].shape[:2]

        # 1. Try OpenCV VideoWriter with MP4
        if cv2 is not None:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(video_path, fourcc, self.fps, (W, H))
                for f in frames:
                    # OpenCV expects BGR
                    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[2] == 3 else f
                    writer.write(bgr)
                writer.release()
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    print(f"[MapVideoRenderer] Reconstructed map video saved: {video_path}")
                    return video_path
            except Exception as e:
                print(f"[MapVideoRenderer] cv2 video write failed: {e}")

        # 2. Try ImageIO
        if imageio is not None:
            try:
                imageio.mimwrite(video_path, frames, fps=self.fps)
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    print(f"[MapVideoRenderer] Reconstructed map video saved via imageio: {video_path}")
                    return video_path
            except Exception as e:
                print(f"[MapVideoRenderer] imageio mp4 write failed: {e}")

        # 3. Fallback to animated GIF
        gif_path = os.path.join(self.output_dir, fallback_gif)
        try:
            pil_frames = [Image.fromarray(f) for f in frames]
            duration_ms = int(1000 / self.fps)
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=True,
                duration=duration_ms,
                loop=0
            )
            print(f"[MapVideoRenderer] Reconstructed map animation saved (GIF): {gif_path}")
            return gif_path
        except Exception as e:
            print(f"[MapVideoRenderer] GIF fallback failed: {e}")

        return None

