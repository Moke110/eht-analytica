"""Model loading (direct in-process) and tracking execution service."""

from __future__ import annotations

import importlib
import json
import os
import threading
import traceback
from pathlib import Path

import torch

from backend.services.task_manager import task_manager
from backend.services.video_service import get_session
from backend.services.job_config import add_samples, update_sample_field
from backend.utils.config import save_model_name


# Module-level model state
_loaded_model = None             # EHTTracker object (direct in-process)
_loaded_model_name: str | None = None         # e.g. "unet_v3"
_loaded_model_device: str | None = None       # e.g. "cuda" or "cpu"
_loaded_model_display_name: str | None = None # e.g. "EHT Tracker v3"


def get_model_status() -> dict:
    return {
        "loaded": _loaded_model is not None,
        "model_name": _loaded_model_name,
        "model_path": None,  # deprecated, kept for backward compat
        "device": _loaded_model_device,
        "device_info": f"{_loaded_model_display_name or ''} on {_loaded_model_device or 'undefined'}",
    }


def load_model_async(model_name: str) -> str:
    """Load a model directly in-process for the given model name. Returns task_id."""
    global _loaded_model, _loaded_model_name, _loaded_model_device, _loaded_model_display_name

    tid = task_manager.create("model_load")

    def worker():
        global _loaded_model, _loaded_model_name, _loaded_model_device, _loaded_model_display_name
        try:
            task_manager.update(tid, 5, "Reading models.json...")

            project_root = Path(__file__).resolve().parent.parent.parent
            models_json = project_root / "model" / "models.json"

            with open(models_json, "r", encoding="utf-8") as f:
                registry = json.load(f)

            entry = None
            for m in registry.get("models", []):
                if m.get("name") == model_name:
                    entry = m
                    break
            if entry is None:
                available = [m.get("name", "?") for m in registry.get("models", [])]
                raise ValueError(
                    f"Model '{model_name}' not found in registry. Available: {', '.join(available)}"
                )

            display_name = entry.get("display_name", model_name)
            definition_rel = entry["definition"]
            weight_rels = entry.get("weights", [])
            load_fn_name = entry.get("load_function", "load_model")

            if not weight_rels:
                raise ValueError(f"Model '{model_name}' has no 'weights' listed in registry")

            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            task_manager.update(tid, 15, "Importing model module...")

            # Resolve definition path to Python module name
            # e.g. "unet_v3/unet_v3.py" → "model.unet_v3.unet_v3"
            def_stem = definition_rel.replace("/", ".").replace("\\", ".")
            if def_stem.endswith(".py"):
                def_stem = def_stem[:-3]
            module_name = f"model.{def_stem}"
            mod = importlib.import_module(module_name)
            load_fn = getattr(mod, load_fn_name)

            # Resolve weight paths (relative to model/ directory)
            model_dir = os.path.dirname(os.path.abspath(str(models_json)))
            weight_paths = [os.path.join(model_dir, w) for w in weight_rels]

            task_manager.update(tid, 40, f"Loading model weights to {device}...")

            # Unload previous model if any
            if _loaded_model is not None:
                del _loaded_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Load the model
            model = load_fn(weight_paths, device)

            _loaded_model = model
            _loaded_model_name = model_name
            _loaded_model_device = str(device)
            _loaded_model_display_name = display_name

            save_model_name(model_name)
            task_manager.complete(tid, {
                "model_name": display_name,
                "model_path": None,
                "device": str(device),
                "device_info": f"{display_name} on {device}",
            })
        except Exception as e:
            traceback.print_exc()
            task_manager.fail(tid, str(e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return tid


def start_tracking(video_id: str, rois: list[dict], output_folder: str,
                   save_tracked_video: bool = False,
                   save_inferences: bool = False) -> str:
    """Run tracking in background. Returns task_id for SSE streaming."""
    if _loaded_model is None:
        raise ValueError("No model loaded")

    model_name = _loaded_model_name

    sess = get_session(video_id)
    if sess is None:
        raise ValueError(f"Unknown video session: {video_id}")

    from functions.tracker import process_tracking

    # Register samples before tracking starts
    roi_names = [r["name"] for r in rois]
    entries = add_samples(output_folder, sess.video_path, roi_names)
    name_to_entry = {e["roi_name"]: e for e in entries}

    # Attach accession info to each ROI dict for the tracker
    for r in rois:
        e = name_to_entry.get(r["name"], {})
        r["sample_id"] = e.get("id", r["name"])
        r["recording_name"] = os.path.splitext(os.path.basename(sess.video_path))[0]

    tid = task_manager.create("tracking")

    def worker():
        try:
            torch.set_num_threads(torch.get_num_threads())

            def progress_cb(current, total, message):
                pct = (current / total * 100) if total > 0 else 0
                task_manager.update(tid, min(pct, 100), message)
                if task_manager.is_cancelled(tid):
                    raise InterruptedError("Task cancelled")

            # Build inference function — direct in-process model call
            def infer_fn(roi_images):
                """roi_images: list of (H,W) uint8 ndarray → list of (num_peaks,2) ndarray"""
                if not roi_images:
                    return []
                results = []
                for img in roi_images:
                    tensor = torch.as_tensor(img, device=_loaded_model_device,
                                             dtype=torch.float32)
                    with torch.inference_mode():
                        coords = _loaded_model(tensor)  # (num_peaks, 2)
                    results.append(coords.detach().cpu().numpy())
                return results

            result = process_tracking(
                video_path=sess.video_path,
                rois=rois,
                infer_fn=infer_fn,
                output_folder=output_folder,
                progress_callback=progress_cb,
                save_tracked_video=save_tracked_video,
                save_inferences=save_inferences,
                model_name=model_name,
            )

            if result.get("success"):
                # Compose tracked video from temp JSON (post-tracking, frame-aligned)
                json_path = result.get("tracked_frames_json")
                if json_path and os.path.isfile(json_path):
                    from functions.tracker import compose_tracked_video_from_json
                    try:
                        video_rel = compose_tracked_video_from_json(json_path)
                        result["tracked_video"] = video_rel
                    except Exception as _ve:
                        print(f"[tracking_service] Failed to compose tracked video: {_ve}")
                    finally:
                        try:
                            os.remove(json_path)
                        except OSError:
                            pass

                for rel_path in result.get("saved_files", []):
                    filename = os.path.basename(rel_path)
                    parts = filename.split("_", 2)
                    if len(parts) >= 3:
                        aid = parts[0]
                        roi_name = parts[2].rsplit("_length.csv", 1)[0]
                        update_sample_field(output_folder, aid, "length_csv", rel_path)
                task_manager.complete(tid, result)
            else:
                task_manager.fail(tid, result.get("message", "Tracking failed"))
        except InterruptedError:
            pass
        except Exception as e:
            traceback.print_exc()
            task_manager.fail(tid, str(e))
        finally:
            # Clean up orphaned temp JSON if tracking wrote one but errored later
            if os.path.isdir(output_folder):
                import fnmatch
                for f in os.listdir(output_folder):
                    if fnmatch.fnmatch(f, '._*_tracked_frames.json'):
                        try:
                            os.remove(os.path.join(output_folder, f))
                        except OSError:
                            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return tid
