"""
ROI Selector Module
===================

This module provides Region of Interest (ROI) selection tools for EHT Analytica.

Classes and Functions
--------------------
(Add your ROI selector classes and functions here)

Example Usage
-------------
>>> from EhtAnalytica.ROIselector import ROISelector
>>> # Use ROI selection functionality

"""

from .video_processor import VideoProcessor, process_video_threaded
from .define_roi import ROIDrawCanvas, enable_roi_drawing, ROI

__all__ = ['VideoProcessor', 'process_video_threaded', 'ROIDrawCanvas', 'enable_roi_drawing', 'ROI']
