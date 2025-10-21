import wx
import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
if __name__ != "__main__":
    from ..selector_n_tracker.video_processor import process_video_threaded
    from ..selector_n_tracker.define_roi import enable_roi_drawing
    from ..selector_n_tracker.track import process_tracking
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from EhtAnalytica.selector_n_tracker.video_processor import process_video_threaded
    from EhtAnalytica.selector_n_tracker.define_roi import enable_roi_drawing
    from EhtAnalytica.selector_n_tracker.track import process_tracking


class TrackPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_video_path = None
        self.current_metadata = None
        self.current_first_frame = None
        self.progress_dialog = None
        self.output_folder_path = None
        self.roi_canvas = None  # ROI drawing canvas
        
        # Track model and recording state
        self.has_recording = False
        self.has_roi_model = False
        self.has_track_model = False
        self.tracker_model = None  # Loaded tracker model instance
        
        self.init_ui()
        # Attempt to load previously saved track model (non-blocking)
        try:
            self._load_saved_track_model_async()
        except Exception:
            pass
        
    def init_ui(self):
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        left_panel = self.create_left_panel()
        main_sizer.Add(left_panel, 0, wx.ALL | wx.EXPAND, 5)
        right_panel = self.create_right_panel()
        main_sizer.Add(right_panel, 1, wx.ALL | wx.EXPAND, 5)
        self.SetSizer(main_sizer)
        
    def create_left_panel(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 1. recording_dir_txt: non-interactable text
        self.recording_dir_txt = wx.StaticText(panel, label="No recording selected", size=wx.Size(300, 80), style=wx.ALIGN_CENTER_VERTICAL | wx.ST_NO_AUTORESIZE)
        self.recording_dir_txt.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.recording_dir_txt.SetBackgroundColour(wx.Colour(240, 240, 240))
        sizer.Add(self.recording_dir_txt, 0, wx.ALL | wx.EXPAND, 5)
        
        # 2. select_rec_btn: button
        self.select_rec_btn = wx.Button(panel, label="Select Recording", size=wx.Size(300, 60))
        self.select_rec_btn.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.select_rec_btn.Bind(wx.EVT_BUTTON, self.on_select_recording)
        sizer.Add(self.select_rec_btn, 0, wx.ALL | wx.EXPAND, 5)
        
        sizer.AddSpacer(20)
        
        # 4. roi_model_txt: non-interactable text
        self.roi_model_txt = wx.StaticText(panel, label="No ROI model selected", size=wx.Size(300, 50), style=wx.ALIGN_CENTER_VERTICAL | wx.ST_NO_AUTORESIZE)
        self.roi_model_txt.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.roi_model_txt.SetBackgroundColour(wx.Colour(240, 240, 240))
        sizer.Add(self.roi_model_txt, 0, wx.ALL | wx.EXPAND, 5)
        
        # Row with buttons 5 and 6
        button_row_1 = wx.BoxSizer(wx.HORIZONTAL)
        
        # 5. select_roi_model_btn: button
        self.select_roi_model_btn = wx.Button(panel, label="Select ROI model", size=wx.Size(145, 60))
        self.select_roi_model_btn.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.select_roi_model_btn.Bind(wx.EVT_BUTTON, self.on_select_roi_model)
        button_row_1.Add(self.select_roi_model_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        # 6. auto_roi_btn: button
        self.auto_roi_btn = wx.Button(panel, label="Auto ROI", size=wx.Size(145, 60))
        self.auto_roi_btn.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.auto_roi_btn.Bind(wx.EVT_BUTTON, self.on_auto_roi)
        self.auto_roi_btn.Enable(False)  # Disabled initially
        button_row_1.Add(self.auto_roi_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(button_row_1, 0, wx.EXPAND)
        
        # 7. track_model_txt: non-interactable text
        self.track_model_txt = wx.StaticText(panel, label="No track model selected", size=wx.Size(300, 50), style=wx.ALIGN_CENTER_VERTICAL | wx.ST_NO_AUTORESIZE)
        self.track_model_txt.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.track_model_txt.SetBackgroundColour(wx.Colour(240, 240, 240))
        sizer.Add(self.track_model_txt, 0, wx.ALL | wx.EXPAND, 5)
        
        # Row with buttons 8 and 9
        button_row_2 = wx.BoxSizer(wx.HORIZONTAL)
        
        # 8. select_track_model_btn: button
        self.select_track_model_btn = wx.Button(panel, label="Select track model", size=wx.Size(145, 60))
        self.select_track_model_btn.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.select_track_model_btn.Bind(wx.EVT_BUTTON, self.on_select_track_model)
        button_row_2.Add(self.select_track_model_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        # 9. track_btn: button
        self.track_btn = wx.Button(panel, label="Track", size=wx.Size(145, 60))
        self.track_btn.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.track_btn.Bind(wx.EVT_BUTTON, self.on_track)
        self.track_btn.Enable(False)  # Disabled initially
        button_row_2.Add(self.track_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(button_row_2, 0, wx.EXPAND)
        
        # 10. output_folder_btn: button
        self.output_folder_btn = wx.Button(panel, label="Open Output Folder", size=wx.Size(300, 60))
        self.output_folder_btn.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.output_folder_btn.Bind(wx.EVT_BUTTON, self.on_open_output)
        self.output_folder_btn.Enable(False)  # Disabled initially
        sizer.Add(self.output_folder_btn, 0, wx.ALL | wx.EXPAND, 5)
        
        sizer.AddStretchSpacer(1)
        
        panel.SetSizer(sizer)
        return panel
    
    def create_right_panel(self):
        panel = wx.Panel(self)
        panel.SetBackgroundColour(wx.WHITE)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 3. display_canvas: interactable canvas
        self.display_canvas = wx.Panel(panel)
        self.display_canvas.SetBackgroundColour(wx.WHITE)
        sizer.Add(self.display_canvas, 1, wx.ALL | wx.EXPAND, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    # Event handlers
    def on_select_recording(self, event):
        wildcard = "Video files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Select Recording File", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
            self.recording_dir_txt.SetLabel(path)
            print(f"Selected recording: {path}")
            
            # Start processing video in background thread
            self.process_video(path)
        dialog.Destroy()
    
    def process_video(self, video_path):
        """Process video in background thread with progress dialog."""
        # Show progress dialog
        self.progress_dialog = wx.ProgressDialog(
            "Importing Recording",
            "Initializing...",
            maximum=100,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
        )
        
        def progress_callback(percent, message):
            """Update progress dialog from worker thread."""
            wx.CallAfter(self._update_progress, percent, message)
        
        def completion_callback(metadata, first_frame, error=None):
            """Handle completion from worker thread."""
            wx.CallAfter(self._on_video_processed, metadata, first_frame, error)
        
        # Start processing in background thread
        process_video_threaded(video_path, progress_callback, completion_callback)
    
    def _update_progress(self, percent, message):
        """Update progress dialog (called from main thread)."""
        if self.progress_dialog:
            self.progress_dialog.Update(int(percent), message)
    
    def _on_video_processed(self, metadata, first_frame, error=None):
        """Handle video processing completion (called from main thread)."""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.Destroy()
            self.progress_dialog = None
        
        if error:
            wx.MessageBox(f"Error processing video: {error}", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        if metadata is None or first_frame is None:
            wx.MessageBox("Failed to process video", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Store metadata and first frame
        self.current_metadata = metadata
        self.current_first_frame = first_frame
        self.current_video_path = metadata['path']
        # Update displayed recording label to reflect the (possibly fixed) file
        try:
            self.recording_dir_txt.SetLabel(self.current_video_path)
        except Exception:
            pass
        
        # Create output folder
        video_path = metadata['path']
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_folder_path = os.path.join(video_dir, f"{video_name}_analytics")
        
        try:
            os.makedirs(self.output_folder_path, exist_ok=True)
            print(f"Created output folder: {self.output_folder_path}")
        except Exception as e:
            wx.MessageBox(f"Warning: Could not create output folder: {e}", "Warning", wx.OK | wx.ICON_WARNING)
        
        # Mark that recording is loaded
        self.has_recording = True
        
        # Enable output_folder_btn
        self.output_folder_btn.Enable(True)
        
        # Update button states based on model availability
        self.update_button_states()
        
        # Enable ROI drawing on canvas
        self.enable_roi_drawing()
        
        # Show video information dialog
        info_message = (
            f"Video Information:\n\n"
            f"Framerate: {metadata['fps']:.2f} fps\n"
            f"Total Frames: {metadata['frame_count']}\n"
            f"Duration: {metadata['duration']:.2f} seconds\n"
            f"Resolution: {metadata['width']}x{metadata['height']}\n\n"
            f"Output folder created: {self.output_folder_path}\n\n"
            f"You can now draw ROIs on the canvas by clicking and dragging.\n\n"
            f"Click OK to proceed."
        )
        
        wx.MessageBox(info_message, "Video Information", wx.OK | wx.ICON_INFORMATION)
    
    def display_frame(self, frame):
        """Display a frame in the canvas."""
        if frame is None:
            return
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to wxPython bitmap
        height, width = frame_rgb.shape[:2]
        image = wx.Image(width, height, frame_rgb.tobytes())
        
        # Scale to fit canvas
        canvas_size = self.display_canvas.GetSize()
        scale = min(canvas_size.width / width, canvas_size.height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        if new_width > 0 and new_height > 0:
            image = image.Scale(new_width, new_height, wx.IMAGE_QUALITY_HIGH)
        
        bitmap = wx.Bitmap(image)
        
        # Create static bitmap to display
        # For wxPython 4.2+, convert Bitmap to BitmapBundle
        if hasattr(wx, 'BitmapBundle'):
            bitmap_bundle: wx.BitmapBundle = wx.BitmapBundle.FromBitmap(bitmap)  # type: ignore
        else:
            bitmap_bundle: wx.BitmapBundle = bitmap  # type: ignore
        
        if hasattr(self, 'frame_display'):
            self.frame_display.SetBitmap(bitmap_bundle)
        else:
            self.frame_display = wx.StaticBitmap(self.display_canvas, bitmap=bitmap_bundle)
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.frame_display, 1, wx.ALL | wx.ALIGN_CENTER, 5)
            self.display_canvas.SetSizer(sizer)
        
        self.display_canvas.Layout()
        self.display_canvas.Refresh()
    
    def enable_roi_drawing(self):
        """Enable ROI drawing on the display canvas."""
        if self.current_first_frame is None or self.current_metadata is None:
            return
        
        # Get original frame size
        original_size = (self.current_metadata['width'], self.current_metadata['height'])
        
        # Get recording name from path
        if self.current_video_path:
            recording_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
        else:
            recording_name = "recording"
        
        # Enable ROI drawing canvas
        self.roi_canvas = enable_roi_drawing(
            self.display_canvas, 
            self.current_first_frame, 
            original_size,
            recording_name
        )
        
        print("ROI drawing enabled. Click and drag to draw rectangles.")
    
    def get_rois(self):
        """Get all drawn ROIs in pixel coordinates."""
        if self.roi_canvas:
            return self.roi_canvas.get_all_rois()
        return []
    
    def update_button_states(self):
        """Update button enabled/disabled states based on current state."""
        # auto_roi_btn: enabled only if recording AND ROI model are loaded
        self.auto_roi_btn.Enable(self.has_recording and self.has_roi_model)
        
        # track_btn: enabled only if recording AND track model are loaded
        self.track_btn.Enable(self.has_recording and self.has_track_model)
        
        print(f"Button states updated - Recording: {self.has_recording}, "
              f"ROI Model: {self.has_roi_model}, Track Model: {self.has_track_model}")
    
    def on_select_roi_model(self, event):
        wildcard = "Model files (*.pth;*.pt;*.h5;*.onnx)|*.pth;*.pt;*.h5;*.onnx|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Select ROI Model File", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
            self.roi_model_txt.SetLabel(path)
            self.has_roi_model = True
            print(f"Selected ROI model: {path}")
            
            # Update button states
            self.update_button_states()
        dialog.Destroy()
    
    def on_auto_roi(self, event):
        print("Auto ROI started")
        
        # Print current ROIs for debugging
        rois = self.get_rois()
        if rois:
            print(f"Current ROIs ({len(rois)}):")
            for roi in rois:
                print(f"  {roi}")
        else:
            print("No ROIs defined yet")
        
        # TODO: Implement auto ROI detection logic
    
    def on_select_track_model(self, event):
        """Select a TorchScript .pt file containing the track model."""
        wildcard = "TorchScript model (*.pt)|*.pt|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Select Track Model (.pt)", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            model_file = dialog.GetPath()

            # Show loading dialog and run in thread
            loading_dialog = wx.ProgressDialog(
                "Loading Track Model",
                "Loading track model, please wait...",
                maximum=100,
                parent=self,
                style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
            )
            loading_dialog.Pulse()

            # Run model loading in a separate thread
            import threading

            def load_thread():
                try:
                    self.load_track_model_pt(model_file)
                finally:
                    wx.CallAfter(loading_dialog.Destroy)

            thread = threading.Thread(target=load_thread)
            thread.daemon = True
            thread.start()

        dialog.Destroy()

    def load_track_model_pt(self, model_file):
        """Load a TorchScript .pt model file and set as tracker_model."""
        try:
            # Lazy import torch
            try:
                import torch
            except Exception as e:
                wx.CallAfter(wx.MessageBox, f"Torch not available: {e}", "Error", wx.OK | wx.ICON_ERROR)
                return

            # Load scripted model
            model = torch.jit.load(model_file, map_location='cpu')
            model.eval()
            self.tracker_model = model

            def update_gui():
                self.has_track_model = True
                self.track_model_txt.SetLabel(os.path.basename(model_file))
                self.update_button_states()

            wx.CallAfter(update_gui)

            # Do not show a blocking success message box so the user can continue interacting.
            print(f"Loaded TorchScript model: {model_file}")
            # Persist selected model path
            try:
                self._save_track_model_path(model_file)
            except Exception:
                pass

        except Exception as e:
            error_msg = str(e)
            wx.CallAfter(wx.MessageBox, f"Error loading TorchScript model: {error_msg}", "Error", wx.OK | wx.ICON_ERROR)
            print(f"Error loading TorchScript model: {error_msg}")
            import traceback
            traceback.print_exc()

    # --- model path persistence helpers ---
    def _config_file_path(self):
        """Return path to config/model_path.json in the repo root; create folder if needed."""
        try:
            repo_root = Path(__file__).resolve().parents[3]
        except Exception:
            repo_root = Path.cwd()
        config_dir = repo_root / 'config'
        os.makedirs(str(config_dir), exist_ok=True)
        return str(config_dir / 'model_path.json')

    def _save_track_model_path(self, model_path):
        """Save the model_path to config/model_path.json as {'track_model': path}."""
        cfg = {'track_model': model_path}
        try:
            cfg_path = self._config_file_path()
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            print(f"Saved track model path to config: {cfg_path}")
        except Exception as e:
            print(f"Warning: could not save model path: {e}")

    def _load_saved_track_model_path(self):
        """Return saved path or None if not present."""
        try:
            cfg_path = self._config_file_path()
            if not os.path.exists(cfg_path):
                return None
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            return cfg.get('track_model')
        except Exception:
            return None

    def _load_saved_track_model_async(self):
        """If a saved model path exists, attempt to load it in a background thread."""
        saved = self._load_saved_track_model_path()
        if not saved:
            return

        # Only proceed if the path exists on disk
        if not os.path.exists(saved):
            print(f"Saved track model path not found: {saved}")
            return

        import threading

        def worker():
            try:
                # If it's a TorchScript file
                if os.path.isfile(saved) and saved.lower().endswith('.pt'):
                    self.load_track_model_pt(saved)
                elif os.path.isdir(saved):
                    # Look for a .pt inside the directory
                    for entry in os.listdir(saved):
                        if entry.lower().endswith('.pt'):
                            candidate = os.path.join(saved, entry)
                            self.load_track_model_pt(candidate)
                            return
                    print(f"No .pt model found in saved directory: {saved}")
                else:
                    print(f"Unsupported saved model path type: {saved}")
            except Exception as e:
                print(f"Error loading saved track model: {e}")

        t = threading.Thread(target=worker, daemon=True)
        t.start()
    
    def on_track(self, event):
        """Start tracking process."""
        # Validate prerequisites
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            wx.MessageBox("No video recording selected.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        if not self.tracker_model:
            wx.MessageBox("No track model loaded.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Get ROIs
        rois = self.get_rois()
        if not rois:
            wx.MessageBox("No ROIs defined. Please draw ROIs first.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        if not self.output_folder_path or not os.path.exists(self.output_folder_path):
            wx.MessageBox("Output folder does not exist.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        print(f"Starting tracking with {len(rois)} ROI(s)...")
        
        # Get total frames for progress
        cap = cv2.VideoCapture(self.current_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Create progress dialog
        progress_dialog = wx.ProgressDialog(
            "Tracking Progress",
            "Starting tracking...",
            maximum=total_frames,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME
        )
        
        # Progress callback for updating dialog
        def progress_callback(current_frame, total_frames, message):
            wx.CallAfter(progress_dialog.Update, current_frame, message)
        
        # Run tracking in a separate thread
        import threading
        
        def tracking_thread():
            try:
                result = process_tracking(
                    video_path=self.current_video_path,
                    rois=rois,
                    tracker_model=self.tracker_model,
                    output_folder=self.output_folder_path,
                    progress_callback=progress_callback
                )
                
                # Close progress dialog and show result
                wx.CallAfter(progress_dialog.Destroy)
                
                if result['success']:
                    wx.CallAfter(
                        wx.MessageBox,
                        result['message'],
                        "Tracking Complete",
                        wx.OK | wx.ICON_INFORMATION
                    )
                    print("Tracking completed successfully!")
                else:
                    wx.CallAfter(
                        wx.MessageBox,
                        result['message'],
                        "Tracking Error",
                        wx.OK | wx.ICON_ERROR
                    )
                    print(f"Tracking error: {result['message']}")
                    
            except Exception as e:
                wx.CallAfter(progress_dialog.Destroy)
                import traceback
                error_msg = f"Unexpected error during tracking:\n{str(e)}\n\n{traceback.format_exc()}"
                wx.CallAfter(
                    wx.MessageBox,
                    error_msg,
                    "Error",
                    wx.OK | wx.ICON_ERROR
                )
                print(error_msg)
        
        thread = threading.Thread(target=tracking_thread)
        thread.daemon = True
        thread.start()
    
    def on_open_output(self, event):
        """Open the output folder in file explorer."""
        if not self.output_folder_path or not os.path.exists(self.output_folder_path):
            wx.MessageBox("Output folder does not exist.", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                os.startfile(self.output_folder_path)
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', self.output_folder_path])
            else:
                subprocess.Popen(['xdg-open', self.output_folder_path])
            print(f"Opened output folder: {self.output_folder_path}")
        except Exception as e:
            wx.MessageBox(f"Could not open folder: {e}", "Error", wx.OK | wx.ICON_ERROR)
