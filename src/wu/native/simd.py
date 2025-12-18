"""
Python bindings to native SIMD functions.

Uses ctypes to call compiled C code. Falls back to NumPy
if native library is not available.
"""

import ctypes
import platform
import sys
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Library handle
_lib: Optional[ctypes.CDLL] = None
_available = False


def _find_library() -> Optional[Path]:
    """Find the native library based on platform."""
    if platform.system() == "Windows":
        lib_name = "wu_simd.dll"
    elif platform.system() == "Darwin":
        lib_name = "libwu_simd.dylib"
    else:
        lib_name = "libwu_simd.so"

    # Search paths
    search_paths = [
        Path(__file__).parent / lib_name,
        Path(__file__).parent / "build" / lib_name,
        Path(__file__).parent.parent.parent.parent / "build" / lib_name,
        Path.cwd() / lib_name,
        Path.cwd() / "build" / lib_name,
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def _load_library() -> bool:
    """Load the native library."""
    global _lib, _available

    lib_path = _find_library()
    if lib_path is None:
        return False

    try:
        _lib = ctypes.CDLL(str(lib_path))

        # Set up function signatures
        _lib.wu_get_simd_caps.restype = ctypes.c_int
        _lib.wu_get_simd_caps.argtypes = []

        _lib.wu_dot_product_f32.restype = ctypes.c_double
        _lib.wu_dot_product_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t
        ]

        _lib.wu_dot_product_f64.restype = ctypes.c_double
        _lib.wu_dot_product_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t
        ]

        _lib.wu_euclidean_distance_f32.restype = ctypes.c_double
        _lib.wu_euclidean_distance_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t
        ]

        _lib.wu_normalize_f32.restype = ctypes.c_double
        _lib.wu_normalize_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t
        ]

        _lib.wu_normalize_f64.restype = ctypes.c_double
        _lib.wu_normalize_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t
        ]

        _lib.wu_variance_f64.restype = ctypes.c_double
        _lib.wu_variance_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t
        ]

        _lib.wu_sobel_3x3.restype = None
        _lib.wu_sobel_3x3.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int
        ]

        _lib.wu_compute_blockiness.restype = ctypes.c_double
        _lib.wu_compute_blockiness.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]

        _available = True
        return True

    except Exception as e:
        _lib = None
        _available = False
        return False


# Try to load library on import
_load_library()


def is_available() -> bool:
    """Check if native SIMD library is available."""
    return _available


def get_simd_caps() -> int:
    """
    Get SIMD capability flags.

    Returns:
        Bitmask: 1=SSE2, 2=AVX, 4=AVX2, 8=AVX512, 16=NEON
    """
    if _lib is not None:
        return _lib.wu_get_simd_caps()
    return 0


def dot_product_f32(a: 'np.ndarray', b: 'np.ndarray') -> float:
    """
    Compute dot product of two float32 arrays using SIMD.

    Args:
        a: First array (float32)
        b: Second array (float32)

    Returns:
        Dot product value
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    if len(a) != len(b):
        raise ValueError("Arrays must have same length")

    if _lib is not None:
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return _lib.wu_dot_product_f32(a_ptr, b_ptr, len(a))

    # NumPy fallback
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)))


def dot_product_f64(a: 'np.ndarray', b: 'np.ndarray') -> float:
    """
    Compute dot product of two float64 arrays using SIMD.

    Args:
        a: First array (float64)
        b: Second array (float64)

    Returns:
        Dot product value
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError("Arrays must have same length")

    if _lib is not None:
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return _lib.wu_dot_product_f64(a_ptr, b_ptr, len(a))

    # NumPy fallback
    return float(np.dot(a, b))


def euclidean_distance_f32(a: 'np.ndarray', b: 'np.ndarray') -> float:
    """
    Compute Euclidean distance between two float32 arrays.

    Args:
        a: First array
        b: Second array

    Returns:
        Euclidean distance
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    if _lib is not None:
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return _lib.wu_euclidean_distance_f32(a_ptr, b_ptr, len(a))

    # NumPy fallback
    return float(np.linalg.norm(a - b))


def normalize_f32(arr: 'np.ndarray') -> float:
    """
    Normalize array to unit length in-place.

    Args:
        arr: Array to normalize (modified in-place)

    Returns:
        Original norm (length) of array
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    arr = np.ascontiguousarray(arr, dtype=np.float32)

    if _lib is not None:
        arr_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return _lib.wu_normalize_f32(arr_ptr, len(arr))

    # NumPy fallback
    norm = float(np.linalg.norm(arr))
    if norm > 1e-10:
        arr /= norm
    return norm


def normalize_f64(arr: 'np.ndarray') -> float:
    """
    Normalize array to unit length in-place.

    Args:
        arr: Array to normalize (modified in-place)

    Returns:
        Original norm (length) of array
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    arr = np.ascontiguousarray(arr, dtype=np.float64)

    if _lib is not None:
        arr_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return _lib.wu_normalize_f64(arr_ptr, len(arr))

    # NumPy fallback
    norm = float(np.linalg.norm(arr))
    if norm > 1e-10:
        arr /= norm
    return norm


def variance_f64(arr: 'np.ndarray') -> float:
    """
    Compute variance of array.

    Args:
        arr: Input array

    Returns:
        Variance
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    arr = np.ascontiguousarray(arr.ravel(), dtype=np.float64)

    if _lib is not None:
        arr_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return _lib.wu_variance_f64(arr_ptr, len(arr))

    # NumPy fallback
    return float(np.var(arr))


def sobel_3x3(image: 'np.ndarray') -> tuple:
    """
    Apply 3x3 Sobel filter for gradient computation.

    Args:
        image: 2D grayscale image (float64)

    Returns:
        Tuple of (gx, gy) gradient arrays
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    image = np.ascontiguousarray(image, dtype=np.float64)
    height, width = image.shape

    gx = np.zeros_like(image)
    gy = np.zeros_like(image)

    if _lib is not None:
        img_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        gx_ptr = gx.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        gy_ptr = gy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _lib.wu_sobel_3x3(img_ptr, gx_ptr, gy_ptr, width, height)
        return gx, gy

    # NumPy fallback using scipy-style convolution
    from scipy.signal import convolve2d

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64) / 8.0

    gx = convolve2d(image, sobel_x, mode='same', boundary='symm')
    gy = convolve2d(image, sobel_y, mode='same', boundary='symm')

    return gx, gy


def compute_blockiness(
    image: 'np.ndarray',
    x_offset: int,
    y_offset: int,
    block_size: int = 8
) -> float:
    """
    Compute blockiness measure for JPEG grid detection.

    Args:
        image: 2D grayscale image (float64)
        x_offset: Grid X offset (0-7)
        y_offset: Grid Y offset (0-7)
        block_size: Block size (typically 8)

    Returns:
        Blockiness score
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    image = np.ascontiguousarray(image, dtype=np.float64)
    height, width = image.shape

    if _lib is not None:
        img_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return _lib.wu_compute_blockiness(
            img_ptr, width, height, x_offset, y_offset, block_size
        )

    # NumPy fallback
    total_diff = 0.0
    count = 0

    # Vertical boundaries
    for x in range(x_offset, width - 1, block_size):
        diff = image[:, x] - image[:, x + 1]
        total_diff += np.sum(diff ** 2)
        count += height

    # Horizontal boundaries
    for y in range(y_offset, height - 1, block_size):
        diff = image[y, :] - image[y + 1, :]
        total_diff += np.sum(diff ** 2)
        count += width

    return total_diff / count if count > 0 else 0.0


# Convenience function to print SIMD info
def print_simd_info():
    """Print SIMD capability information."""
    caps = get_simd_caps()
    print(f"Native SIMD library: {'available' if is_available() else 'not found'}")
    print(f"SIMD capabilities: {caps}")
    print(f"  SSE2:   {bool(caps & 1)}")
    print(f"  AVX:    {bool(caps & 2)}")
    print(f"  AVX2:   {bool(caps & 4)}")
    print(f"  AVX512: {bool(caps & 8)}")
    print(f"  NEON:   {bool(caps & 16)}")
