# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

_root = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(_root))

a = Analysis(
    [str(_root / 'backend' / 'launcher.py')],
    pathex=[],
    binaries=[],
    datas=[
        (str(_root / 'frontend' / 'dist'), 'frontend_dist'),
        (str(_root / 'functions'), 'functions'),
        (str(_root / 'training'), 'training'),
        (str(_root / 'backend'), 'backend'),
    ],
    hiddenimports=[
        'uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.protocols',
        'uvicorn.lifespan', 'uvicorn.lifespan.on',
        'fastapi', 'starlette', 'pydantic',
        'cv2', 'numpy', 'pandas', 'torch',
        'PIL', 'tkinter', 'tkinter.filedialog',
        'model.unet_v3.unet_v3',
        'training.src.dataset.inference_collector',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['wx', 'matplotlib', 'jedi', 'IPython', 'ipykernel',
              'albumentations', 'skimage', 'sklearn', 'tensorboard'],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EHT_Analytica',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    icon=str(_root / 'img' / 'logo.ico'),
    console=False,
    disable_windowed_tracked=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='EHT_Analytica',
)
