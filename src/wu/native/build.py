#!/usr/bin/env python3
"""
Build script for Wu native SIMD library.

Usage:
    python build.py          # Auto-detect compiler and build
    python build.py --clean  # Clean build artifacts
    python build.py --test   # Build and run tests

Requirements:
    - GCC (Linux/macOS) or MSVC (Windows)
    - CPU with AVX2 support for best performance
"""

import subprocess
import platform
import sys
import shutil
from pathlib import Path


def find_compiler():
    """Find available C compiler."""
    if platform.system() == "Windows":
        # Check for MSVC
        try:
            result = subprocess.run(
                ["where", "cl.exe"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return "msvc"
        except FileNotFoundError:
            pass

        # Check for GCC (MinGW)
        try:
            result = subprocess.run(
                ["where", "gcc.exe"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return "gcc"
        except FileNotFoundError:
            pass

        # Check for Clang
        try:
            result = subprocess.run(
                ["where", "clang.exe"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return "clang"
        except FileNotFoundError:
            pass

    else:
        # Unix-like systems
        for compiler in ["gcc", "clang", "cc"]:
            if shutil.which(compiler):
                return compiler

    return None


def build_windows_msvc(src_dir: Path, out_dir: Path):
    """Build with MSVC on Windows."""
    out_dir.mkdir(exist_ok=True)
    dll_path = out_dir / "wu_simd.dll"

    cmd = [
        "cl.exe",
        "/O2",           # Optimize for speed
        "/arch:AVX2",    # Enable AVX2
        "/LD",           # Create DLL
        "/Fe:" + str(dll_path),
        str(src_dir / "wu_simd.c"),
    ]

    print(f"Building with MSVC: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(src_dir))
    return result.returncode == 0


def build_windows_gcc(src_dir: Path, out_dir: Path):
    """Build with GCC (MinGW) on Windows."""
    out_dir.mkdir(exist_ok=True)
    dll_path = out_dir / "wu_simd.dll"

    cmd = [
        "gcc",
        "-O3",           # Optimize for speed
        "-mavx2",        # Enable AVX2
        "-mfma",         # Enable FMA
        "-shared",       # Create shared library
        "-o", str(dll_path),
        str(src_dir / "wu_simd.c"),
    ]

    print(f"Building with GCC: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def build_unix(src_dir: Path, out_dir: Path, compiler: str):
    """Build on Linux/macOS."""
    out_dir.mkdir(exist_ok=True)

    if platform.system() == "Darwin":
        lib_name = "libwu_simd.dylib"
        # Check for ARM64 Mac
        if platform.machine() == "arm64":
            arch_flags = []  # NEON is always available on ARM64
        else:
            arch_flags = ["-mavx2", "-mfma"]
    else:
        lib_name = "libwu_simd.so"
        arch_flags = ["-mavx2", "-mfma"]

    lib_path = out_dir / lib_name

    cmd = [
        compiler,
        "-O3",           # Optimize for speed
        *arch_flags,     # Architecture-specific flags
        "-shared",       # Create shared library
        "-fPIC",         # Position-independent code
        "-o", str(lib_path),
        str(src_dir / "wu_simd.c"),
        "-lm",           # Link math library
    ]

    print(f"Building with {compiler}: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def build():
    """Main build function."""
    src_dir = Path(__file__).parent
    out_dir = src_dir

    compiler = find_compiler()
    if compiler is None:
        print("ERROR: No C compiler found!")
        print("Install GCC, Clang, or MSVC to build native library.")
        return False

    print(f"Found compiler: {compiler}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    if platform.system() == "Windows":
        if compiler == "msvc":
            return build_windows_msvc(src_dir, out_dir)
        else:
            return build_windows_gcc(src_dir, out_dir)
    else:
        return build_unix(src_dir, out_dir, compiler)


def clean():
    """Remove build artifacts."""
    src_dir = Path(__file__).parent

    patterns = [
        "*.dll", "*.so", "*.dylib",
        "*.obj", "*.o", "*.lib", "*.exp",
    ]

    for pattern in patterns:
        for f in src_dir.glob(pattern):
            print(f"Removing: {f}")
            f.unlink()

    print("Clean complete.")


def test():
    """Build and run basic tests."""
    if not build():
        print("Build failed!")
        return False

    print("\nRunning tests...")

    # Import and test
    try:
        import numpy as np
        from . import simd

        if not simd.is_available():
            print("WARNING: Native library not loaded, using fallback")

        simd.print_simd_info()

        # Test dot product
        a = np.random.rand(1000).astype(np.float32)
        b = np.random.rand(1000).astype(np.float32)

        result_native = simd.dot_product_f32(a, b)
        result_numpy = float(np.dot(a.astype(np.float64), b.astype(np.float64)))

        print(f"\nDot product test:")
        print(f"  Native: {result_native}")
        print(f"  NumPy:  {result_numpy}")
        print(f"  Diff:   {abs(result_native - result_numpy)}")

        if abs(result_native - result_numpy) < 1e-4:
            print("  PASS")
        else:
            print("  FAIL")
            return False

        # Test variance
        arr = np.random.rand(1000).astype(np.float64)
        var_native = simd.variance_f64(arr)
        var_numpy = float(np.var(arr))

        print(f"\nVariance test:")
        print(f"  Native: {var_native}")
        print(f"  NumPy:  {var_numpy}")
        print(f"  Diff:   {abs(var_native - var_numpy)}")

        if abs(var_native - var_numpy) < 1e-10:
            print("  PASS")
        else:
            print("  FAIL")
            return False

        print("\nAll tests passed!")
        return True

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Wu native SIMD library")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--test", action="store_true", help="Build and run tests")
    args = parser.parse_args()

    if args.clean:
        clean()
    elif args.test:
        test()
    else:
        if build():
            print("\nBuild successful!")
            print("Library will be automatically loaded by wu.native.simd")
        else:
            print("\nBuild failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
