#!/usr/bin/env python3
"""
Build script for creating standalone Wu CLI executable.

This script builds a standalone executable using PyInstaller that
can be distributed without requiring Python installation.

Usage:
    python build_cli.py
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Build the standalone executable."""
    print("Building Wu CLI executable...")
    print("=" * 60)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("ERROR: PyInstaller is not installed!")
        print("Install with: pip install pyinstaller")
        sys.exit(1)
    
    # Check if spec file exists
    spec_file = Path("wu.spec")
    if not spec_file.exists():
        print(f"ERROR: {spec_file} not found!")
        sys.exit(1)
    
    # Build the executable
    print(f"\nBuilding with spec file: {spec_file}")
    print("-" * 60)
    
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        exe_path = Path("dist") / "wu.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print("\n" + "=" * 60)
            print("[SUCCESS] Build successful!")
            print(f"  Executable: {exe_path}")
            print(f"  Size: {size_mb:.1f} MB")
            print("\nYou can now distribute wu.exe as a standalone CLI tool.")
            print("Test it with: dist\\wu.exe --help")
        else:
            print("\nWARNING: Build completed but executable not found!")
            sys.exit(1)
    else:
        print("\nERROR: Build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

