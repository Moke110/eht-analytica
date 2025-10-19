"""
Main Window Module
==================

This module provides the MainPanel class which serves as the main application window
for EHT Analytica, containing a notebook with multiple panels including the TrackPanel.
"""

import wx
from .track_panel import TrackPanel


class MainPanel(wx.Frame):
    """
    Main application window with notebook interface.
    
    The window has a default size of 1920x1080 pixels and contains a notebook
    with multiple pages including the tracking panel.
    
    Parameters
    ----------
    parent : wx.Window or None
        Parent window (typically None for main frame)
    title : str, optional
        Window title (default: "EHT Analytica")
    """
    
    def __init__(self, parent=None, title="EHT Analytica"):
        """Initialize the main window."""
        super().__init__(parent, title=title, size=(1920, 1080))
        
        # Center the window on screen
        self.Centre()
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Set up the user interface with notebook and panels."""
        # Create main panel
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create notebook (tabbed interface)
        self.notebook = wx.Notebook(panel)
        
        # Add Track Panel as the first page
        self.track_panel = TrackPanel(self.notebook)
        self.notebook.AddPage(self.track_panel, "Track")
        
        # Add placeholder for additional panels/pages
        # You can add more pages here in the future
        # Example:
        # self.analysis_panel = AnalysisPanel(self.notebook)
        # self.notebook.AddPage(self.analysis_panel, "Analysis")
        
        sizer.Add(self.notebook, 1, wx.ALL | wx.EXPAND, 5)
        panel.SetSizer(sizer)
        
        # Set icon (optional - uncomment if you have an icon file)
        # try:
        #     icon = wx.Icon("path/to/icon.ico", wx.BITMAP_TYPE_ICO)
        #     self.SetIcon(icon)
        # except:
        #     pass
