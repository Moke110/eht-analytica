"""
Main Entry Point for EHT Analytica GUI
=======================================

This module serves as the main entry point for the EHT Analytica GUI application.
It initializes the wxPython application and creates the main window.

Usage
-----
Run this module directly to start the GUI application:

    python -m EhtAnalytica.main

Or from the command line after installation:

    eht-analytica
"""

import wx
import sys
import os

# Add parent directory to path for imports
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from EhtAnalytica.gui.main_window import MainPanel
else:
    from .gui.main_window import MainPanel


def main():
    """
    Main entry point for the EHT Analytica GUI application.
    
    This function initializes the wxPython application, creates the main window,
    and starts the event loop.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    # Create the wxPython application
    app = wx.App(False)
    
    # Create and show the main window
    frame = MainPanel(title="EHT Analytica - Track & Analyze")
    frame.Show()
    
    # Set the main frame as the top window
    app.SetTopWindow(frame)
    
    # Start the event loop
    app.MainLoop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
