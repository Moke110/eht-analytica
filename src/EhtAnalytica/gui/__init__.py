"""
GUI Module
==========

This module provides the graphical user interface components for EHT Analytica.

Components
----------
MainPanel : class
    Main application window with notebook interface
TrackPanel : class
    Tracking panel with controls and display area

Example Usage
-------------
>>> from EhtAnalytica.gui import MainPanel
>>> import wx
>>> app = wx.App()
>>> frame = MainPanel()
>>> frame.Show()
>>> app.MainLoop()
"""

from .main_window import MainPanel
from .track_panel import TrackPanel

__all__ = [
    'MainPanel',
    'TrackPanel',
]
