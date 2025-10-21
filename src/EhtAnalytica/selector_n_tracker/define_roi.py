import wx
import numpy as np


class ROI:
    """Represents a single Region of Interest."""
    
    def __init__(self, rect, color, name, original_frame_size, canvas_size):
        """
        Initialize an ROI.
        
        Parameters
        ----------
        rect : wx.Rect
            Rectangle in canvas coordinates
        color : wx.Colour
            Color for the ROI
        name : str
            Name of the ROI
        original_frame_size : tuple
            (width, height) of the original frame
        canvas_size : tuple
            (width, height) of the canvas
        """
        self.rect = rect  # Canvas coordinates
        self.color = color
        self.name = name
        self.original_frame_size = original_frame_size
        self.canvas_size = canvas_size
        
        # Calculate pixel coordinates in original frame
        self.pixel_rect = self._convert_to_pixel_space()
    
    def _convert_to_pixel_space(self):
        """Convert canvas coordinates to original frame pixel coordinates."""
        orig_width, orig_height = self.original_frame_size
        canvas_width, canvas_height = self.canvas_size
        
        # Calculate scale factor (same as used in display_frame)
        scale = min(canvas_width / orig_width, canvas_height / orig_height)
        
        # Calculate offset to center the image
        scaled_width = int(orig_width * scale)
        scaled_height = int(orig_height * scale)
        offset_x = (canvas_width - scaled_width) // 2
        offset_y = (canvas_height - scaled_height) // 2
        
        # Convert canvas coordinates to scaled coordinates
        x = self.rect.x - offset_x
        y = self.rect.y - offset_y
        w = self.rect.width
        h = self.rect.height
        
        # Convert to original frame pixel space
        pixel_x = int(x / scale)
        pixel_y = int(y / scale)
        pixel_w = int(w / scale)
        pixel_h = int(h / scale)
        
        # Clamp to frame boundaries
        pixel_x = max(0, min(pixel_x, orig_width))
        pixel_y = max(0, min(pixel_y, orig_height))
        pixel_w = max(0, min(pixel_w, orig_width - pixel_x))
        pixel_h = max(0, min(pixel_h, orig_height - pixel_y))
        
        return wx.Rect(pixel_x, pixel_y, pixel_w, pixel_h)
    
    def get_pixel_coordinates(self):
        """Get ROI coordinates in original frame pixel space."""
        return {
            'name': self.name,
            'x': self.pixel_rect.x,
            'y': self.pixel_rect.y,
            'width': self.pixel_rect.width,
            'height': self.pixel_rect.height,
            'color': (self.color.Red(), self.color.Green(), self.color.Blue())
        }


class ROIDrawCanvas(wx.Panel):
    """Canvas for drawing ROIs on a video frame."""
    
    # Predefined colors for ROIs (bright colors only, no black or dark colors)
    COLORS = [
        wx.Colour(255, 0, 0),      # Red
        wx.Colour(0, 255, 0),      # Green
        wx.Colour(0, 128, 255),    # Light Blue
        wx.Colour(255, 255, 0),    # Yellow
        wx.Colour(255, 0, 255),    # Magenta
        wx.Colour(0, 255, 255),    # Cyan
        wx.Colour(255, 128, 0),    # Orange
        wx.Colour(255, 128, 255),  # Light Purple
        wx.Colour(128, 255, 128),  # Light Green
        wx.Colour(255, 192, 203),  # Pink
    ]
    
    def __init__(self, parent, frame, original_frame_size, recording_name="recording"):
        """
        Initialize the ROI drawing canvas.
        
        Parameters
        ----------
        parent : wx.Window
            Parent window
        frame : numpy.ndarray
            Video frame to display
        original_frame_size : tuple
            (width, height) of the original frame
        recording_name : str, optional
            Name of the recording for ROI naming
        """
        super().__init__(parent)
        self.SetBackgroundColour(wx.WHITE)
        try:
            # Recommended for buffered painting
            self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        except Exception:
            pass
        # Reduce flicker and improve paint performance
        try:
            self.SetDoubleBuffered(True)
        except Exception:
            pass
        
        self.frame = frame
        self.original_frame_size = original_frame_size
        self.recording_name = recording_name
        self.bitmap = None
        self.rois = []  # List of ROI objects
        self.roi_controls = []  # List of (text_ctrl, delete_btn) tuples
        
        # Drawing state
        self.drawing = False
        self.start_pos = None
        self.current_rect = None
        
        # Bind events
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_mouse_up)
        self.Bind(wx.EVT_MOTION, self.on_mouse_move)
        self.Bind(wx.EVT_SIZE, self.on_size)
        
        # Set the frame
        self.set_frame(frame)
    
    def set_frame(self, frame):
        """Set the video frame to display."""
        if frame is None:
            return
        
        import cv2
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to wxPython bitmap
        height, width = frame_rgb.shape[:2]
        image = wx.Image(width, height, frame_rgb.tobytes())
        
        # Scale to fit canvas
        canvas_size = self.GetSize()
        if canvas_size.width <= 1 or canvas_size.height <= 1:
            # Canvas not laid out yet; try again after layout
            self.frame = frame
            try:
                wx.CallAfter(self.set_frame, frame)
            except Exception:
                pass
        else:
            scale = min(canvas_size.width / width, canvas_size.height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if new_width > 0 and new_height > 0:
                image = image.Scale(new_width, new_height, wx.IMAGE_QUALITY_HIGH)
        
        self.bitmap = wx.Bitmap(image)
        self.frame = frame
        self.Refresh()
    
    def on_size(self, event):
        """Handle window resize."""
        if self.frame is not None:
            self.set_frame(self.frame)
        event.Skip()
    
    def on_paint(self, event):
        """Paint the canvas with frame and ROIs."""
        # Buffered paint to avoid flicker
        dc_cls = getattr(wx, 'AutoBufferedPaintDC', wx.BufferedPaintDC)
        dc = dc_cls(self)
        # Ensure clear uses our background colour
        try:
            dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        except Exception:
            pass
        dc.Clear()
        
        if self.bitmap:
            # Center the bitmap
            bmp_width = self.bitmap.GetWidth()
            bmp_height = self.bitmap.GetHeight()
            canvas_size = self.GetSize()
            x = (canvas_size.width - bmp_width) // 2
            y = (canvas_size.height - bmp_height) // 2
            dc.DrawBitmap(self.bitmap, x, y)
        
        # Draw existing ROIs
        for roi in self.rois:
            dc.SetPen(wx.Pen(roi.color, 2))
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.DrawRectangle(roi.rect.x, roi.rect.y, roi.rect.width, roi.rect.height)
        
        # Draw current drawing rectangle
        if self.drawing and self.current_rect:
            color = self._get_next_color()
            dc.SetPen(wx.Pen(color, 2, wx.PENSTYLE_DOT))
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.DrawRectangle(self.current_rect.x, self.current_rect.y, 
                           self.current_rect.width, self.current_rect.height)
    
    def on_mouse_down(self, event):
        """Start drawing a rectangle."""
        self.drawing = True
        self.start_pos = event.GetPosition()
        self.current_rect = wx.Rect(self.start_pos.x, self.start_pos.y, 0, 0)
    
    def on_mouse_move(self, event):
        """Update rectangle while drawing."""
        if self.drawing and self.start_pos:
            current_pos = event.GetPosition()
            
            # Calculate rectangle
            x = min(self.start_pos.x, current_pos.x)
            y = min(self.start_pos.y, current_pos.y)
            w = abs(current_pos.x - self.start_pos.x)
            h = abs(current_pos.y - self.start_pos.y)
            
            self.current_rect = wx.Rect(x, y, w, h)
            self.Refresh()
    
    def on_mouse_up(self, event):
        """Finish drawing a rectangle and create ROI."""
        if self.drawing and self.current_rect:
            # Only create ROI if rectangle has some size
            if self.current_rect.width > 10 and self.current_rect.height > 10:
                self.create_roi(self.current_rect)
            
            self.drawing = False
            self.start_pos = None
            self.current_rect = None
            self.Refresh()
    
    def create_roi(self, rect):
        """Create a new ROI from a rectangle."""
        color = self._get_next_color()
        name = f"{self.recording_name}_{len(self.rois) + 1}"
        
        canvas_size = self.GetSize()
        roi = ROI(rect, color, name, self.original_frame_size, 
                  (canvas_size.width, canvas_size.height))
        
        self.rois.append(roi)
        
        # Create label and delete button
        self.create_roi_controls(roi, len(self.rois) - 1)
        
        print(f"Created {name}: Canvas {rect}, Pixel {roi.pixel_rect}")
        print(f"  Pixel coordinates: {roi.get_pixel_coordinates()}")
    
    def create_roi_controls(self, roi, roi_index):
        """Create text label and delete button for an ROI."""
        # Create a small panel to hold the controls
        control_panel = wx.Panel(self)
        control_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Editable text control with ROI name
        text_ctrl = wx.TextCtrl(control_panel, value=roi.name, style=wx.TE_PROCESS_ENTER)
        text_ctrl.SetForegroundColour(roi.color)
        text_ctrl.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        # Commit name on Enter or when focus is lost
        text_ctrl.Bind(wx.EVT_TEXT_ENTER, lambda evt, idx=roi_index: self._on_roi_name_enter(evt, idx))
        text_ctrl.Bind(wx.EVT_KILL_FOCUS, lambda evt, idx=roi_index: self._on_roi_name_focus_lost(evt, idx))
        control_sizer.Add(text_ctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        
        # Delete button
        # Use a clear delete label (multiplication sign)
        delete_btn = wx.Button(control_panel, label="\u00D7", size=wx.Size(25, 25))
        delete_btn.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        delete_btn.Bind(wx.EVT_BUTTON, lambda evt, idx=roi_index: self.delete_roi(idx))
        control_sizer.Add(delete_btn, 0, wx.ALIGN_CENTER_VERTICAL)

        control_panel.SetSizer(control_sizer)
        control_panel.Fit()

        # Position the control panel above the ROI
        rect = roi.rect
        control_panel.SetPosition(wx.Point(rect.x, rect.y - 30))

        # Store the text control, delete button and panel
        self.roi_controls.append((text_ctrl, delete_btn, control_panel))
    
    def delete_roi(self, roi_index):
        """Delete an ROI and its controls."""
        if 0 <= roi_index < len(self.rois):
            # Remove controls
            if roi_index < len(self.roi_controls):
                text_ctrl, btn, panel = self.roi_controls[roi_index]
                panel.Destroy()
                self.roi_controls.pop(roi_index)
            
            # Remove ROI
            roi = self.rois.pop(roi_index)
            print(f"Deleted {roi.name}")
            
            # Update control indices for remaining ROIs
            for i in range(roi_index, len(self.rois)):
                if i < len(self.roi_controls):
                    text_ctrl, btn, panel = self.roi_controls[i]
                    btn.Unbind(wx.EVT_BUTTON)
                    btn.Bind(wx.EVT_BUTTON, lambda evt, idx=i: self.delete_roi(idx))
                    # Rebind name events to correct index
                    text_ctrl.Unbind(wx.EVT_TEXT_ENTER)
                    text_ctrl.Unbind(wx.EVT_KILL_FOCUS)
                    text_ctrl.Bind(wx.EVT_TEXT_ENTER, lambda evt, idx=i: self._on_roi_name_enter(evt, idx))
                    text_ctrl.Bind(wx.EVT_KILL_FOCUS, lambda evt, idx=i: self._on_roi_name_focus_lost(evt, idx))
            
            self.Refresh()
    
    def _get_next_color(self):
        """Get the next color for a new ROI."""
        return self.COLORS[len(self.rois) % len(self.COLORS)]
    
    def get_all_rois(self):
        """Get all ROIs in pixel coordinates."""
        return [roi.get_pixel_coordinates() for roi in self.rois]
    
    def clear_rois(self):
        """Clear all ROIs."""
        for text, btn, panel in self.roi_controls:
            panel.Destroy()
        
        self.rois.clear()
        self.roi_controls.clear()
        self.Refresh()

    # --- ROI name editing handlers ---
    def _on_roi_name_enter(self, event, roi_index):
        """Handle Enter key in ROI name edit control."""
        try:
            self._commit_roi_name(roi_index)
        finally:
            event.Skip()

    def _on_roi_name_focus_lost(self, event, roi_index):
        """Handle focus lost for ROI name edit control."""
        try:
            self._commit_roi_name(roi_index)
        finally:
            event.Skip()

    def _commit_roi_name(self, roi_index):
        """Validate and commit ROI name change from control to ROI object."""
        if not (0 <= roi_index < len(self.rois)):
            return

        # Get control and new name
        try:
            text_ctrl, _, _ = self.roi_controls[roi_index]
            new_name = text_ctrl.GetValue().strip()
        except Exception:
            return

        if not new_name:
            wx.MessageBox("ROI name cannot be empty.", "Invalid Name", wx.OK | wx.ICON_WARNING)
            # revert to previous name
            text_ctrl.SetValue(self.rois[roi_index].name)
            return

        # Check for duplicates
        for i, roi in enumerate(self.rois):
            if i != roi_index and roi.name == new_name:
                wx.MessageBox("ROI name already exists. Choose a different name.", "Duplicate Name", wx.OK | wx.ICON_WARNING)
                text_ctrl.SetValue(self.rois[roi_index].name)
                return

        # Commit new name
        old_name = self.rois[roi_index].name
        self.rois[roi_index].name = new_name
        print(f"Renamed ROI '{old_name}' -> '{new_name}'")
        # Refresh canvas to redraw any labels
        self.Refresh()


def enable_roi_drawing(parent_panel, frame, original_frame_size, recording_name="recording"):
    """
    Enable ROI drawing on a panel.
    
    Parameters
    ----------
    parent_panel : wx.Panel
        Parent panel to add the ROI canvas to
    frame : numpy.ndarray
        Video frame to display
    original_frame_size : tuple
        (width, height) of the original frame
    recording_name : str, optional
        Name of the recording for ROI naming
        
    Returns
    -------
    ROIDrawCanvas
        The ROI drawing canvas
    """
    # Clear existing children
    for child in parent_panel.GetChildren():
        child.Destroy()
    
    # Create ROI canvas
    roi_canvas = ROIDrawCanvas(parent_panel, frame, original_frame_size, recording_name)
    
    # Add to parent sizer
    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(roi_canvas, 1, wx.EXPAND)
    parent_panel.SetSizer(sizer)
    # Force layout and refresh so the canvas gets a real size before first paint
    try:
        parent_panel.Layout()
        if parent_panel.GetParent() is not None:
            parent_panel.GetParent().Layout()
        parent_panel.Refresh()
    except Exception:
        pass
    # Re-apply frame after layout to ensure correct scaling
    try:
        wx.CallAfter(roi_canvas.set_frame, frame)
    except Exception:
        pass
    return roi_canvas
