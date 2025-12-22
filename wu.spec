# -*- mode: python ; coding: utf-8 -*-
import sys
import os

# Add src to path so we can import wu during build
src_path = os.path.abspath('src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Try to collect modules dynamically, fallback to explicit list
try:
    from PyInstaller.utils.hooks import collect_submodules, collect_data_files
    wu_modules = collect_submodules('wu')
    wu_data = collect_data_files('wu', includes=['**/*.dll', '**/*.so', '**/*.dylib'])
except (ImportError, ModuleNotFoundError):
    # Fallback if PyInstaller hooks not available or wu not importable
    wu_modules = []
    wu_data = []

# Hidden imports for all wu submodules
hiddenimports = [
    'wu',
    'wu.cli',
    'wu.analyzer',
    'wu.aggregator',
    'wu.state',
    'wu.report',
    'wu.reference',
    'wu.dimensions',
    'wu.dimensions.metadata',
    'wu.dimensions.c2pa',
    'wu.dimensions.visual',
    'wu.dimensions.gps',
    'wu.dimensions.enf',
    'wu.dimensions.copymove',
    'wu.dimensions.prnu',
    'wu.dimensions.blockgrid',
    'wu.dimensions.lighting',
    'wu.dimensions.audio',
    'wu.dimensions.thumbnail',
    'wu.dimensions.geometry',
    'wu.dimensions.quantization',
    'wu.dimensions.aigen',
    'wu.dimensions.devices',
    'wu.video',
    'wu.video.analyzer',
    'wu.video.bitstream',
    'wu.video.box_parser',
    'wu.video.cavlc',
    'wu.video.h264_headers',
    'wu.video.h264_inter',
    'wu.video.h264_intra',
    'wu.video.h264_slice',
    'wu.video.nal_extractor',
    'wu.video.decoders',
    'wu.video.decoders.h264',
    'wu.video.decoders.mjpeg',
    'wu.native',
    'wu.native.simd',
    'click',
    'exifread',
    'PIL',
    'PIL.Image',
    'reportlab',
    'jinja2',
] + wu_modules

# Collect data files (templates, etc.)
datas = []
if os.path.exists('src/wu/report/templates'):
    datas.append(('src/wu/report/templates', 'wu/report/templates'))

# Include native DLL only if it exists (not present on CI runners)
native_binaries = []
if os.path.exists('src/wu/native/wu_simd.dll'):
    native_binaries.append(('src/wu/native/wu_simd.dll', 'wu/native'))

a = Analysis(
    ['wu_launcher.py'],
    pathex=['src'],
    binaries=native_binaries + (wu_data if wu_data else []),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'],
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
    name='wu',
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
