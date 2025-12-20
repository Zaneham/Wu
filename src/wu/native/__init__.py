"""
Native SIMD-optimized functions for Wu forensics.

This module provides Python bindings to C functions that use
AVX2/SSE/NEON SIMD instructions for significant speedups on
computationally intensive operations.

Usage:
    from wu.native import simd



    # Use optimized functions
    result = simd.dot_product(array1, array2)

If the native library is not available, functions fall back to
pure Python/NumPy implementations.
"""

from .simd import (
    is_available,
    get_simd_caps,
    dot_product_f32,
    dot_product_f64,
    euclidean_distance_f32,
    normalize_f32,
    normalize_f64,
    variance_f64,
    sobel_3x3,
    compute_blockiness,
)

__all__ = [
    "is_available",
    "get_simd_caps",
    "dot_product_f32",
    "dot_product_f64",
    "euclidean_distance_f32",
    "normalize_f32",
    "normalize_f64",
    "variance_f64",
    "sobel_3x3",
    "compute_blockiness",
]
