# -*- mode: python ; coding: utf-8 -*-
# Onefile spec for the EHT Analytica Setup wizard (the Installer).
# The payload manifest (manifest.json) must sit next to this spec file —
# build/make_release.py writes it before invoking PyInstaller.

from pathlib import Path

_dir = Path(SPECPATH).resolve()
_root = _dir.parent.parent

_icon = _root / 'img' / 'logo.ico'
if not _icon.exists():
    _icon = _root / 'frontend' / 'public' / 'logo.ico'

_manifest = _dir / 'manifest.json'
_datas = [(str(_manifest), '.')] if _manifest.exists() else []

a = Analysis(
    [str(_dir / 'wizard.py')],
    pathex=[str(_dir)],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        'tkinter', 'tkinter.ttk', 'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    excludes=[
        # Keep the Installer small: nothing from the app stack ships here.
        'numpy', 'pandas', 'cv2', 'torch', 'scipy', 'psutil', 'PIL',
        'fastapi', 'uvicorn', 'pydantic', 'matplotlib',
    ],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='EHT_Analytica-Setup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(_icon) if _icon.exists() else None,
)
