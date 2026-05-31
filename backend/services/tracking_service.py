"""Model loading and tracking execution service."""

from __future__ import annotations

import os
import threading

from backend.services.task_manager import task_manager
from backend.services.video_service import get_session
from backend.utils.config import save_model_path


# Module-level model state
loaded_model = None
model_device: str | None = None
model_name: str | None = None
model_path: str | None = None
device_info: str | None = None


def get_model_status() -> dict:
    return {
        "loaded": loaded_model is not None,
        "model_name": model_name,
        "model_path": model_path,
        "device": model_device,
        "device_info": device_info,
    }


def _resolve_device_info(device: "torch.device") -> str:
    import torch
    try:
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            total_mb = props.total_memory / (1024 ** 2)
            return f"{props.name} ({total_mb:.0f} MB)"
        else:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            cpu_name = "CPU"
            try:
                import platform
                cpu_name = platform.processor() or "CPU"
                if not cpu_name or cpu_name == "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel":
                    import subprocess
                    result = subprocess.run(
                        ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=5
                    )
                    lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and l.strip() != "Name"]
                    if lines:
                        cpu_name = lines[0]
            except Exception:
                pass
            return f"{cpu_name} ({total_gb:.1f} GB RAM)"
    except Exception:
        return str(device)


def load_model_async(path: str) -> str:
    """Load a TorchScript model in background. Returns task_id."""
    global loaded_model, model_device, model_name, model_path, device_info

    tid = task_manager.create("model_load")

    def worker():
        global loaded_model, model_device, model_name, model_path, device_info
        try:
            import torch

            task_manager.update(tid, 10, "Detecting device...")
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_str)

            task_manager.update(tid, 30, "Loading model...")
            model = torch.jit.load(path, map_location=device)
            model.eval()

            loaded_model = model
            model_device = device_str
            model_name = os.path.basename(path)
            model_path = path
            device_info = _resolve_device_info(device)

            save_model_path(path)
            task_manager.complete(tid, {
                "model_name": model_name,
                "model_path": path,
                "device": model_device,
                "device_info": device_info,
            })
        except Exception as e:
            task_manager.fail(tid, str(e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return tid


def start_tracking(video_id: str, rois: list[dict], output_folder: str) -> str:
    """Run tracking in background. Returns task_id for SSE streaming."""
    global loaded_model

    if loaded_model is None:
        raise ValueError("No model loaded")

    sess = get_session(video_id)
    if sess is None:
        raise ValueError(f"Unknown video session: {video_id}")

    from functions.tracker import process_tracking

    tid = task_manager.create("tracking")

    def worker():
        try:
            # Use all available CPU threads for inference
            import torch
            torch.set_num_threads(torch.get_num_threads())

            def progress_cb(current, total, message):
                pct = (current / total * 100) if total > 0 else 0
                task_manager.update(tid, min(pct, 100), message)
                if task_manager.is_cancelled(tid):
                    raise InterruptedError("Task cancelled")

            result = process_tracking(
                video_path=sess.video_path,
                rois=rois,
                tracker_model=loaded_model,
                output_folder=output_folder,
                progress_callback=progress_cb,
                total_frames_hint=sess.valid_frame_count,
                frame_buffer=sess.frame_buffer if sess.frame_buffer else None,
            )

            if result.get("success"):
                task_manager.complete(tid, result)
            else:
                task_manager.fail(tid, result.get("message", "Tracking failed"))
        except InterruptedError:
            pass  # cancel already handled
        except Exception as e:
            task_manager.fail(tid, str(e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return tid
