# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui\\main_app.py'],
    pathex=[],
    binaries=[],
    datas=[('gui/QtGui.ui', 'gui'), ('src/facenet_keras_weights.h5', 'src'), ('src/encodings/encodings.pkl', 'src/encodings'), ('database/face_recognition.db', 'database')],
    hiddenimports=[],
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
    [],
    exclude_binaries=True,
    name='main_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_app',
)
