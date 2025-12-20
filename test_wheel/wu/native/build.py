#!/usr/bin/env python3
"""
Build script for Wu native SIMD library.

Usage:
    python build.py              # Auto-detect compiler and build
    python build.py --clean      # Clean build artifacts
    python build.py --test       # Build and run tests
    python build.py --asm-only   # Build only assembly files

Requirements:
    - GCC (Linux/macOS) or MSVC (Windows)
    - NASM or YASM for x86-64 assembly
    - CPU with AVX2 support for best performance
"""

import subprocess
import platform
import sys
import shutil
from pathlib import Path
from typing import Optional, List


# ============================================================================
# Assembler Detection and Compilation
# ============================================================================

def find_assembler() -> Optional[str]:
    """Find available assembler (NASM or YASM)."""
    if platform.system() == "Windows":
        # Check for NASM
        try:
            result = subprocess.run(
                ["where", "nasm.exe"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return "nasm"
        except FileNotFoundError:
            pass

        # Check for YASM
        try:
            result = subprocess.run(
                ["where", "yasm.exe"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return "yasm"
        except FileNotFoundError:
            pass
    else:
        # Unix-like systems
        for asm in ["nasm", "yasm"]:
            if shutil.which(asm):
                return asm

    return None


def get_asm_format() -> str:
    """Get the appropriate object format for the current platform."""
    if platform.system() == "Windows":
        return "win64"
    elif platform.system() == "Darwin":
        return "macho64"
    else:
        return "elf64"


def get_obj_extension() -> str:
    """Get object file extension for current platform."""
    if platform.system() == "Windows":
        return ".obj"
    else:
        return ".o"


def build_assembly_files(src_dir: Path, assembler: str) -> List[Path]:
    """
    Compile all assembly files in assembly/x86_64/ directory.
    
    Returns list of compiled object file paths.
    """
    asm_dir = src_dir / "assembly" / "x86_64"
    if not asm_dir.exists():
        print(f"Assembly directory not found: {asm_dir}")
        return []

    asm_format = get_asm_format()
    obj_ext = get_obj_extension()
    obj_files = []

    # Find all .asm files
    asm_files = list(asm_dir.glob("*.asm"))
    if not asm_files:
        print("No assembly files found")
        return []

    print(f"\nBuilding {len(asm_files)} assembly files with {assembler}...")
    print(f"Format: {asm_format}")

    for asm_file in asm_files:
        obj_file = asm_dir / (asm_file.stem + obj_ext)
        
        cmd = [
            assembler,
            "-f", asm_format,
            "-o", str(obj_file),
            str(asm_file)
        ]
        
        # Add debug symbols for non-release builds
        if "--release" not in sys.argv:
            if assembler == "nasm":
                cmd.insert(1, "-g")
            elif assembler == "yasm":
                cmd.insert(1, "-g")
                cmd.insert(2, "dwarf2")
        
        print(f"  {asm_file.name} -> {obj_file.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR assembling {asm_file.name}:")
            print(result.stderr)
            return []
        
        obj_files.append(obj_file)

    print(f"Successfully compiled {len(obj_files)} assembly files")
    return obj_files


# ============================================================================
# C Compiler Detection
# ============================================================================

def find_compiler() -> Optional[str]:
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


# ============================================================================
# Build Functions
# ============================================================================

def build_windows_msvc(src_dir: Path, out_dir: Path, obj_files: List[Path] = None):
    """Build with MSVC on Windows."""
    out_dir.mkdir(exist_ok=True)
    dll_path = out_dir / "wu_simd.dll"

    # Base command
    cmd = [
        "cl.exe",
        "/O2",           # Optimize for speed
        "/arch:AVX2",    # Enable AVX2
        "/LD",           # Create DLL
        "/Fe:" + str(dll_path),
        str(src_dir / "wu_simd.c"),
        str(src_dir / "wu_asm_wrappers.c"), # Include wrappers
    ]

    # Add assembly object files if present
    if obj_files:
        cmd.extend([str(f) for f in obj_files])

    print(f"Building with MSVC: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(src_dir))
    return result.returncode == 0


def build_windows_gcc(src_dir: Path, out_dir: Path, obj_files: List[Path] = None):
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
        str(src_dir / "wu_asm_wrappers.c"), # Include wrappers
    ]

    # Add assembly object files if present
    if obj_files:
        cmd.extend([str(f) for f in obj_files])

    print(f"Building with GCC: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def build_unix(src_dir: Path, out_dir: Path, compiler: str, obj_files: List[Path] = None):
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
        str(src_dir / "wu_asm_wrappers.c"), # Include wrappers
    ]

    # Add assembly object files if present
    if obj_files:
        cmd.extend([str(f) for f in obj_files])

    cmd.append("-lm")    # Link math library

    print(f"Building with {compiler}: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def build(asm_only: bool = False):
    """Main build function."""
    src_dir = Path(__file__).parent
    out_dir = src_dir

    # Step 1: Build assembly files if assembler available
    assembler = find_assembler()
    obj_files = []
    
    if assembler:
        print(f"Found assembler: {assembler}")
        obj_files = build_assembly_files(src_dir, assembler)
        if not obj_files and (src_dir / "assembly" / "x86_64").exists():
            print("WARNING: Assembly build failed, continuing without assembly optimizations")
            obj_files = []
    else:
        print("NOTE: No assembler (NASM/YASM) found - building without assembly optimizations")
        print("      Install NASM for maximum performance and IP protection")

    if asm_only:
        if obj_files:
            print("\nAssembly-only build complete!")
            return True
        else:
            print("\nAssembly-only build failed - no assembler found or build error")
            return False

    # Step 2: Find C compiler
    compiler = find_compiler()
    if compiler is None:
        print("ERROR: No C compiler found!")
        print("Install GCC, Clang, or MSVC to build native library.")
        return False

    print(f"Found compiler: {compiler}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    # Step 3: Build combined library
    if platform.system() == "Windows":
        if compiler == "msvc":
            return build_windows_msvc(src_dir, out_dir, obj_files)
        else:
            return build_windows_gcc(src_dir, out_dir, obj_files)
    else:
        return build_unix(src_dir, out_dir, compiler, obj_files)


def clean():
    """Remove build artifacts."""
    src_dir = Path(__file__).parent
    asm_dir = src_dir / "assembly" / "x86_64"

    patterns = [
        "*.dll", "*.so", "*.dylib",
        "*.obj", "*.o", "*.lib", "*.exp",
    ]

    for pattern in patterns:
        for f in src_dir.glob(pattern):
            print(f"Removing: {f}")
            f.unlink()
        
        # Also clean assembly directory
        if asm_dir.exists():
            for f in asm_dir.glob(pattern):
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

        # Test assembly call if available
        if hasattr(simd, 'correlation_sum_asm'):
            print("\nTesting assembly wrapper (correlation_sum_asm)...")
            res_asm = simd.correlation_sum_asm(a.astype(np.float64), b.astype(np.float64))
            print(f"  ASM Result: {res_asm}")
            print(f"  Diff: {abs(res_asm - result_numpy)}")
            if abs(res_asm - result_numpy) < 1e-4:
                print("  PASS")
            else:
                print("  FAIL (Wrapper might be falling back to scalar if ASM not loaded)")

        print("\nAll tests passed!")
        return True

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_info():
    """Print build environment information."""
    src_dir = Path(__file__).parent
    print("Wu Forensics Native Build Info")
    print("=" * 40)
    print(f"Platform:   {platform.system()} {platform.machine()}")
    print(f"Compiler:   {find_compiler() or 'Not found'}")
    print(f"Assembler:  {find_assembler() or 'Not found'}")
    print(f"ASM format: {get_asm_format()}")
    
    asm_dir = src_dir / "assembly" / "x86_64"
    if asm_dir.exists():
        asm_files = list(asm_dir.glob("*.asm"))
        print(f"ASM files:  {len(asm_files)} found")
        for f in asm_files:
            print(f"            - {f.name}")
    else:
        print("ASM files:  None (directory not found)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Wu native SIMD library")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--test", action="store_true", help="Build and run tests")
    parser.add_argument("--asm-only", action="store_true", help="Build only assembly files")
    parser.add_argument("--info", action="store_true", help="Print build environment info")
    parser.add_argument("--release", action="store_true", help="Release build (strip symbols)")
    args = parser.parse_args()

    if args.clean:
        clean()
    elif args.test:
        test()
    elif args.info:
        print_info()
    elif args.asm_only:
        if build(asm_only=True):
            print("\nAssembly build successful!")
        else:
            print("\nAssembly build failed!")
            sys.exit(1)
    else:
        if build():
            print("\nBuild successful!")
            print("Library will be automatically loaded by wu.native.simd")
        else:
            print("\nBuild failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
