"""
EHT Analytica - A Cross-Platform Analytical Application
========================================================

EHT Analytica is a comprehensive analytical toolkit for Windows and macOS,
providing advanced analysis, tracking, and ROI selection capabilities.

Basic Usage
-----------
>>> import EhtAnalytica
>>> print(EhtAnalytica.__version__)
>>> # Access submodules
>>> from EhtAnalytica import analyzer, selector_n_tracker
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Chang Wang"
__email__ = "changwangbj02@gmail.com"
__license__ = "No license specified"
__copyright__ = "none"
__url__ = "https://github.com/Moke110/EHT_Analytica"
__description__ = "A deep learning EHT analysis tool"

# Version info tuple
VERSION = tuple(map(int, __version__.split('.')))

# Import submodules (lazy loading)
import sys
from pathlib import Path

# Add src directory to path if needed
_package_path = Path(__file__).parent
if str(_package_path) not in sys.path:
    sys.path.insert(0, str(_package_path))

# Core module imports
try:
    from . import analyzer
    from . import selector_n_tracker
    from . import gui
    from . import main
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all submodules: {e}", ImportWarning)
    analyzer = None
    selector_n_tracker = None
    gui = None
    main = None

# Public API exports
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "VERSION",
    
    # Submodules
    "analyzer",
    "selector_n_tracker",
    "gui",
    "main",
    
    # Functions
    "get_version",
    "get_info",
]


def get_version():
    """
    Get the current version of EHT Analytica.
    
    Returns
    -------
    str
        Version string in format 'X.Y.Z'
    
    Examples
    --------
    >>> import EhtAnalytica
    >>> EhtAnalytica.get_version()
    '0.1.0'
    """
    return __version__


def get_info():
    """
    Get comprehensive information about the EHT Analytica package.
    
    Returns
    -------
    dict
        Dictionary containing package metadata including version,
        author, description, and platform information.
    
    Examples
    --------
    >>> import EhtAnalytica
    >>> info = EhtAnalytica.get_info()
    >>> print(info['version'])
    '0.1.0'
    """
    import platform
    
    return {
        "name": "EHT Analytica",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": __description__,
        "url": __url__,
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "modules": {
            "analyzer": analyzer is not None,
            "selector_n_tracker": selector_n_tracker is not None,
            "gui": gui is not None,
            "main": main is not None,
        }
    }


# Print welcome message when imported interactively
if hasattr(sys, 'ps1'):
    print(f"EHT Analytica v{__version__} loaded successfully")
    print(f"Platform: {sys.platform}")
    print("Type 'EhtAnalytica.get_info()' for more information")
