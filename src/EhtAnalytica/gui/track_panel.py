import wx
import cv2
import numpy as np
import sys
import os

# Add parent directory to path for imports
if __name__ != "__main__":
    from ..ROIselector.video_processor import process_video_threaded
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from EhtAnalytica.ROIselector.video_processor import process_video_threaded


class TrackPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_video_path = None
        self.current_metadata = None
        self.current_first_frame = None
        self.progress_dialog = None
        self.output_folder_path = None
        self.init_ui()
        
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
        self.recording_dir_txt = wx.StaticText(panel, label="No recording selected", size=(300, 80), style=wx.ALIGN_CENTER_VERTICAL | wx.ST_NO_AUTORESIZE)
        self.recording_dir_txt.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.recording_dir_txt.SetBackgroundColour(wx.Colour(240, 240, 240))
        sizer.Add(self.recording_dir_txt, 0, wx.ALL | wx.EXPAND, 5)
        
        # 2. select_rec_btn: button
        self.select_rec_btn = wx.Button(panel, label="Select Recording", size=(300, 60))
        self.select_rec_btn.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.select_rec_btn.Bind(wx.EVT_BUTTON, self.on_select_recording)
        sizer.Add(self.select_rec_btn, 0, wx.ALL | wx.EXPAND, 5)
        
        sizer.AddSpacer(20)
        
        # 4. roi_model_txt: non-interactable text
        self.roi_model_txt = wx.StaticText(panel, label="No ROI model selected", size=(300, 50), style=wx.ALIGN_CENTER_VERTICAL | wx.ST_NO_AUTORESIZE)
        self.roi_model_txt.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.roi_model_txt.SetBackgroundColour(wx.Colour(240, 240, 240))
        sizer.Add(self.roi_model_txt, 0, wx.ALL | wx.EXPAND, 5)
        
        # Row with buttons 5 and 6
        button_row_1 = wx.BoxSizer(wx.HORIZONTAL)
        
        # 5. select_roi_model_btn: button
        self.select_roi_model_btn = wx.Button(panel, label="Select ROI model", size=(145, 60))
        self.select_roi_model_btn.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.select_roi_model_btn.Bind(wx.EVT_BUTTON, self.on_select_roi_model)
        button_row_1.Add(self.select_roi_model_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        # 6. auto_roi_btn: button
        self.auto_roi_btn = wx.Button(panel, label="Auto ROI", size=(145, 60))
        self.auto_roi_btn.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.auto_roi_btn.Bind(wx.EVT_BUTTON, self.on_auto_roi)
        self.auto_roi_btn.Enable(False)  # Disabled initially
        button_row_1.Add(self.auto_roi_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(button_row_1, 0, wx.EXPAND)
        
        # 7. track_model_txt: non-interactable text
        self.track_model_txt = wx.StaticText(panel, label="No track model selected", size=(300, 50), style=wx.ALIGN_CENTER_VERTICAL | wx.ST_NO_AUTORESIZE)
        self.track_model_txt.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.track_model_txt.SetBackgroundColour(wx.Colour(240, 240, 240))
        sizer.Add(self.track_model_txt, 0, wx.ALL | wx.EXPAND, 5)
        
        # Row with buttons 8 and 9
        button_row_2 = wx.BoxSizer(wx.HORIZONTAL)
        
        # 8. select_track_model_btn: button
        self.select_track_model_btn = wx.Button(panel, label="Select track model", size=(145, 60))
        self.select_track_model_btn.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.select_track_model_btn.Bind(wx.EVT_BUTTON, self.on_select_track_model)
        button_row_2.Add(self.select_track_model_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        # 9. track_btn: button
        self.track_btn = wx.Button(panel, label="Track", size=(145, 60))
        self.track_btn.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.track_btn.Bind(wx.EVT_BUTTON, self.on_track)
        self.track_btn.Enable(False)  # Disabled initially
        button_row_2.Add(self.track_btn, 1, wx.ALL | wx.EXPAND, 5)
        
        sizer.Add(button_row_2, 0, wx.EXPAND)
        
        # 10. output_folder_btn: button
        self.output_folder_btn = wx.Button(panel, label="Open Output Folder", size=(300, 60))
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
        
        # Enable auto_roi_btn and output_folder_btn
        self.auto_roi_btn.Enable(True)
        self.output_folder_btn.Enable(True)
        
        # Display first frame in canvas
        self.display_frame(first_frame)
        
        # Show video information dialog
        info_message = (
            f"Video Information:\n\n"
            f"Framerate: {metadata['fps']:.2f} fps\n"
            f"Total Frames: {metadata['frame_count']}\n"
            f"Duration: {metadata['duration']:.2f} seconds\n"
            f"Resolution: {metadata['width']}x{metadata['height']}\n\n"
            f"Output folder created: {self.output_folder_path}\n\n"
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
        if hasattr(self, 'frame_display'):
            self.frame_display.SetBitmap(bitmap)
        else:
            self.frame_display = wx.StaticBitmap(self.display_canvas, bitmap=bitmap)
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.frame_display, 1, wx.ALL | wx.ALIGN_CENTER, 5)
            self.display_canvas.SetSizer(sizer)
        
        self.display_canvas.Layout()
        self.display_canvas.Refresh()
    
    def on_select_roi_model(self, event):
        wildcard = "Model files (*.pth;*.pt;*.h5;*.onnx)|*.pth;*.pt;*.h5;*.onnx|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Select ROI Model File", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
            self.roi_model_txt.SetLabel(path)
            print(f"Selected ROI model: {path}")
        dialog.Destroy()
    
    def on_auto_roi(self, event):
        print("Auto ROI started")
        # TODO: Implement auto ROI detection logic
    
    def on_select_track_model(self, event):
        wildcard = "Model files (*.pth;*.pt;*.h5;*.onnx)|*.pth;*.pt;*.h5;*.onnx|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Select Track Model File", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
            self.track_model_txt.SetLabel(path)
            print(f"Selected track model: {path}")
        dialog.Destroy()
    
    def on_track(self, event):
        print("Tracking started")
        # TODO: Implement tracking logic
    
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
