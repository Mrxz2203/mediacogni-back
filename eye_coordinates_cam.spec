# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['eye_coordinates_cam.py'],
    pathex=[],
    binaries=[],
    datas=[('models\\face_landmarker_v2.task', 'models'), ('.venv\\Lib\\site-packages\\mediapipe\\tasks\\c\\libmediapipe.dll', 'mediapipe/tasks/c')],
    hiddenimports=['mediapipe.tasks.c'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='eye_coordinates_cam',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
