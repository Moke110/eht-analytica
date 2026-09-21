"""
EHT Analytica - reorganized package
===================================

Top-level package name uses underscore (EHT_Analytica) for Python import compatibility.
Subpackages:
- functions: analysis and tracking utilities
- gui: GUI components
- main: entry point
"""

__version__ = "0.1.0"
__author__ = "Chang Wang"
__email__ = "changwangbj02@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/Moke110/eht_analytica"
__description__ = "A deep learning EHT analysis tool"

VERSION = tuple(map(int, __version__.split('.')))

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "VERSION",
]
