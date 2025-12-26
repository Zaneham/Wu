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
    ]

    # Support PyInstaller bundles
    if hasattr(sys, "_MEIPASS"):
        search_paths.append(Path(sys._MEIPASS) / "wu" / "native" / lib_name)
        search_paths.append(Path(sys._MEIPASS) / "src" / "wu" / "native" / lib_name)

    search_paths.extend([
        Path(__file__).parent.parent.parent.parent / "build" / lib_name,
        Path.cwd() / lib_name,
        Path.cwd() / "build" / lib_name,
    ])

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

        # Assembly wrapper functions
        _lib.wu_similarity_match_asm.restype = ctypes.c_float
        _lib.wu_similarity_match_asm.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]

        # Batch similarity search
        _lib.wu_find_similar_blocks_asm.restype = ctypes.c_int
        _lib.wu_find_similar_blocks_asm.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]


        _lib.wu_correlation_sum_asm.restype = ctypes.c_double
        _lib.wu_correlation_sum_asm.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t
        ]

        _lib.wu_mean_variance_asm.restype = None
        _lib.wu_mean_variance_asm.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

        _lib.wu_find_peak_asm.restype = ctypes.c_double
        _lib.wu_find_peak_asm.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t)
        ]

        _lib.wu_gradient_magnitude_asm.restype = None
        _lib.wu_gradient_magnitude_asm.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double)
        ]

        _lib.wu_blockiness_all_offsets_asm.restype = ctypes.c_int
        _lib.wu_blockiness_all_offsets_asm.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double)
        ]

        _lib.wu_h264_idct_4x4.restype = None
        _lib.wu_h264_idct_4x4.argtypes = [
            ctypes.POINTER(ctypes.c_int16)
        ]

        _lib.wu_h264_filter_6tap.restype = None
        _lib.wu_h264_filter_6tap.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int
        ]

        # Forensic analysis functions
        _lib.wu_qp_scan_horizontal.restype = ctypes.c_int
        _lib.wu_qp_scan_horizontal.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int
        ]

        _lib.wu_qp_frame_stats.restype = None
        _lib.wu_qp_frame_stats.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32)
        ]

        _lib.wu_qp_histogram.restype = None
        _lib.wu_qp_histogram.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32)
        ]

        _lib.wu_qp_frame_discontinuities.restype = ctypes.c_int
        _lib.wu_qp_frame_discontinuities.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int
        ]

        _lib.wu_mv_scan_horizontal.restype = ctypes.c_int
        _lib.wu_mv_scan_horizontal.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int
        ]

        _lib.wu_mv_field_stats.restype = None
        _lib.wu_mv_field_stats.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int
        ]

        _lib.wu_mv_coherence_score.restype = ctypes.c_int
        _lib.wu_mv_coherence_score.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int
        ]

        _lib.wu_mv_detect_outliers.restype = ctypes.c_int
        _lib.wu_mv_detect_outliers.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int
        ]

        _lib.wu_scan_nal_units.restype = ctypes.c_int
        _lib.wu_scan_nal_units.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int
        ]

        _lib.wu_analyse_nal_sequence.restype = ctypes.c_int
        _lib.wu_analyse_nal_sequence.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int
        ]

        _lib.wu_count_epb.restype = ctypes.c_int
        _lib.wu_count_epb.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int
        ]

        _lib.wu_entropy_stats.restype = None
        _lib.wu_entropy_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32)
        ]

        _available = True
        return True

    except Exception:
        # Silently fail, fallback to NumPy
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


# ============================================================================
# ASSEMBLY-ACCELERATED FUNCTIONS
# ============================================================================

def similarity_match_asm(a: 'np.ndarray', b: 'np.ndarray') -> float:
    """Similarity matching using assembly (proprietary algorithm)."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if len(a) != len(b):
        raise ValueError("Arrays must have same length")
    if _lib is not None:
        return _lib.wu_similarity_match_asm(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(a)
        )
    # Fallback
    dot = float(np.dot(a, b))
    norm = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return dot / norm if norm > 1e-10 else 0.0


def correlation_sum_asm(a: 'np.ndarray', b: 'np.ndarray') -> float:
    """Correlation sum using assembly."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("Arrays must have same length")
    if _lib is not None:
        return _lib.wu_correlation_sum_asm(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(a)
        )
    return float(np.dot(a, b))


def mean_variance_asm(data: 'np.ndarray') -> tuple:
    """Compute mean and variance using assembly."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")
    data = np.ascontiguousarray(data.ravel(), dtype=np.float64)
    if _lib is not None:
        mean = ctypes.c_double()
        var = ctypes.c_double()
        _lib.wu_mean_variance_asm(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(data),
            ctypes.byref(mean),
            ctypes.byref(var)
        )
        return mean.value, var.value
    return float(np.mean(data)), float(np.var(data))


def find_peak_asm(data: 'np.ndarray') -> tuple:
    """Find peak value and index using assembly."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")
    data = np.ascontiguousarray(data.ravel(), dtype=np.float64)
    if _lib is not None:
        idx = ctypes.c_size_t()
        peak = _lib.wu_find_peak_asm(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(data),
            ctypes.byref(idx)
        )
        return peak, idx.value
    max_idx = int(np.argmax(data))
    return float(data[max_idx]), max_idx


def gradient_magnitude_asm(gx: 'np.ndarray', gy: 'np.ndarray') -> 'np.ndarray':
    """Compute gradient magnitude using assembly."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")
    gx = np.ascontiguousarray(gx.ravel(), dtype=np.float64)
    gy = np.ascontiguousarray(gy.ravel(), dtype=np.float64)
    if len(gx) != len(gy):
        raise ValueError("Arrays must have same length")
    mag = np.empty(len(gx), dtype=np.float64)
    if _lib is not None:
        _lib.wu_gradient_magnitude_asm(
            gx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            gy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(gx),
            mag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return mag
    return np.sqrt(gx**2 + gy**2)


def blockiness_all_offsets_asm(image: 'np.ndarray', block_size: int = 8) -> tuple:
    """Compute blockiness for all 64 offsets using assembly.
    Returns (best_idx, scores_array) where scores is [8,8].
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")
    image = np.ascontiguousarray(image, dtype=np.float64)
    height, width = image.shape
    scores = np.empty(64, dtype=np.float64)
    if _lib is not None:
        best = _lib.wu_blockiness_all_offsets_asm(
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            width, height, block_size,
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return best, scores.reshape(8, 8)
    # Fallback
    for y in range(8):
        for x in range(8):
            scores[y*8+x] = compute_blockiness(image, x, y, block_size)
    best = int(np.argmin(scores))
    return best, scores.reshape(8, 8)


def find_similar_blocks_asm(features, positions, threshold=0.9, min_distance=10.0, max_matches=10000):
    """
    Find similar blocks using assembly optimization.
    
    Args:
        features: (n_blocks, n_features) float32 array
        positions: (n_blocks, 2) int32 array of x,y coordinates
        threshold: minimum similarity (0-1)
        min_distance: minimum spatial distance
        max_matches: maximum matches to return
        
    Returns:
        matches: (count, 3) array of [index_i, index_j, similarity]
    """
    if not is_available() or not HAS_NUMPY:
        return np.zeros((0, 3), dtype=np.float32)
        
    features = np.ascontiguousarray(features, dtype=np.float32)
    positions = np.ascontiguousarray(positions, dtype=np.int32)
    
    n_blocks = features.shape[0]
    n_features = features.shape[1]
    
    matches_out = np.zeros(max_matches * 3, dtype=np.float32)
    
    if _lib:
        f_ptr = features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_ptr = positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        m_ptr = matches_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # print(f"DEBUG: Calling ASM. features={hex(ctypes.addressof(f_ptr.contents))}, matches={hex(ctypes.addressof(m_ptr.contents))}")
        # print(f"DEBUG: n_blocks={n_blocks}, n_features={n_features}")

        count = _lib.wu_find_similar_blocks_asm(
            f_ptr,
            p_ptr,
            n_blocks,
            n_features,
            ctypes.c_float(threshold),
            ctypes.c_float(min_distance),
            m_ptr,
            max_matches
        )
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        return matches_out[:count*3].reshape(count, 3)
    
    return np.zeros((0, 3), dtype=np.float32)


def h264_idct_4x4(block: 'np.ndarray') -> None:
    """
    Perform H.264 4x4 integer inverse transform.
    Modifies the block in-place.

    Args:
        block: (16,) or (4,4) int16 array
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    block = np.ascontiguousarray(block, dtype=np.int16)
    if block.size != 16:
        raise ValueError("Block must have 16 elements")

    if _lib is not None:
        ptr = block.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        _lib.wu_h264_idct_4x4(ptr)
        return

    # Pure Python IDCT fallback (H.264 spec)
    b = block.ravel()
    tmp = [0] * 16
    # Vertical
    for i in range(4):
        a = int(b[i]) + int(b[i+8])
        bb = int(b[i]) - int(b[i+8])
        c = (int(b[i+4]) >> 1) - int(b[i+12])
        d = int(b[i+4]) + (int(b[i+12]) >> 1)
        tmp[i] = a + d
        tmp[i+4] = bb + c
        tmp[i+8] = bb - c
        tmp[i+12] = a - d
    # Horizontal
    for i in range(4):
        idx = i * 4
        a = tmp[idx] + tmp[idx+2]
        bb = tmp[idx] - tmp[idx+2]
        c = (tmp[idx+1] >> 1) - tmp[idx+3]
        d = tmp[idx+1] + (tmp[idx+3] >> 1)
        b[idx] = (a + d + 32) >> 6
        b[idx+1] = (bb + c + 32) >> 6
        b[idx+2] = (bb - c + 32) >> 6
        b[idx+3] = (a - d + 32) >> 6


def h264_filter_6tap(src: 'np.ndarray', stride: int, dst: 'np.ndarray', width: int, height: int) -> None:
    """
    H.264 6-tap filter for half-pixel interpolation.
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")
    
    src = np.ascontiguousarray(src, dtype=np.uint8)
    dst = np.ascontiguousarray(dst, dtype=np.int16)
    
    if _lib is not None:
        src_ptr = src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        _lib.wu_h264_filter_6tap(src_ptr, stride, dst_ptr, width, height)
        return
        
    # Python fallback
    for y in range(height):
        for x in range(width):
            # Slow scalar fallback for verification
            a = int(src[y * stride + x - 2])
            b = int(src[y * stride + x - 1])
            c = int(src[y * stride + x])
            d = int(src[y * stride + x + 1])
            e = int(src[y * stride + x + 2])
            f = int(src[y * stride + x + 3])
            dst[y * width + x] = a - 5*b + 20*c + 20*d - 5*e + f


# ============================================================================
# Forensic Analysis Functions
# ============================================================================

def qp_scan_horizontal(qp_row: 'np.ndarray', threshold: int = 8, max_out: int = 100):
    """
    Scan a row of QP values for sharp horizontal boundaries.

    Args:
        qp_row: Array of QP values (int8), one per macroblock
        threshold: Minimum delta to flag as boundary (typically 8-12)
        max_out: Maximum number of boundaries to record

    Returns:
        List of (mb_x, delta) tuples for each detected boundary
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    qp_row = np.ascontiguousarray(qp_row.ravel(), dtype=np.int8)
    boundaries = np.zeros(max_out * 2, dtype=np.int32)

    if _lib is not None:
        count = _lib.wu_qp_scan_horizontal(
            qp_row.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            len(qp_row),
            threshold,
            boundaries.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            max_out
        )
        return [(int(boundaries[i*2]), int(boundaries[i*2+1])) for i in range(count)]

    # Fallback
    result = []
    for i in range(len(qp_row) - 1):
        delta = abs(int(qp_row[i]) - int(qp_row[i + 1]))
        if delta >= threshold and len(result) < max_out:
            result.append((i, delta))
    return result


def qp_frame_stats(qp_map: 'np.ndarray', width_mbs: int, height_mbs: int):
    """
    Compute QP statistics for an entire frame.

    Args:
        qp_map: QP values for all macroblocks (row-major)
        width_mbs: Frame width in macroblocks
        height_mbs: Frame height in macroblocks

    Returns:
        Dict with keys: sum, min, max, count
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    qp_map = np.ascontiguousarray(qp_map.ravel(), dtype=np.int8)
    stats = np.zeros(4, dtype=np.int32)

    if _lib is not None:
        _lib.wu_qp_frame_stats(
            qp_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            width_mbs,
            height_mbs,
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        )
    else:
        total = width_mbs * height_mbs
        qp_u8 = qp_map.view(np.uint8)
        stats[0] = int(np.sum(qp_u8[:total]))
        stats[1] = int(np.min(qp_u8[:total]))
        stats[2] = int(np.max(qp_u8[:total]))
        stats[3] = total

    return {
        'sum': int(stats[0]),
        'min': int(stats[1]),
        'max': int(stats[2]),
        'count': int(stats[3])
    }


def qp_histogram(qp_map: 'np.ndarray'):
    """
    Build histogram of QP values (52 bins, 0-51).

    Args:
        qp_map: QP values for all macroblocks

    Returns:
        np.ndarray of shape (52,) with counts
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    qp_map = np.ascontiguousarray(qp_map.ravel(), dtype=np.int8)
    histogram = np.zeros(52, dtype=np.int32)

    if _lib is not None:
        _lib.wu_qp_histogram(
            qp_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            len(qp_map),
            histogram.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        )
    else:
        for qp in qp_map:
            if 0 <= qp <= 51:
                histogram[qp] += 1

    return histogram


def qp_frame_discontinuities(avg_qp_history: 'np.ndarray', threshold: float = 10.0, max_out: int = 100):
    """
    Detect frames with large QP changes from previous frame.

    Args:
        avg_qp_history: Array of average QP per frame
        threshold: Minimum delta to flag (typically 10-15)
        max_out: Maximum discontinuities to record

    Returns:
        List of frame indices where discontinuities occur
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    avg_qp_history = np.ascontiguousarray(avg_qp_history.ravel(), dtype=np.float32)
    discontinuities = np.zeros(max_out, dtype=np.int32)

    if _lib is not None:
        count = _lib.wu_qp_frame_discontinuities(
            avg_qp_history.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(avg_qp_history),
            threshold,
            discontinuities.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            max_out
        )
        return [int(discontinuities[i]) for i in range(count)]

    # Fallback
    result = []
    for i in range(1, len(avg_qp_history)):
        delta = abs(avg_qp_history[i] - avg_qp_history[i - 1])
        if delta >= threshold and len(result) < max_out:
            result.append(i)
    return result


def mv_scan_horizontal(mv_row: 'np.ndarray', threshold_sq: int = 1024, max_out: int = 100):
    """
    Scan a row of motion vectors for spatial discontinuities.

    Args:
        mv_row: Array of MVs as [mvx0, mvy0, mvx1, mvy1, ...] (int16)
        threshold_sq: Squared magnitude threshold (e.g. 1024 = 32^2)
        max_out: Maximum discontinuities to record

    Returns:
        List of (mb_x, magnitude_sq) tuples
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    mv_row = np.ascontiguousarray(mv_row.ravel(), dtype=np.int16)
    width_mbs = len(mv_row) // 2
    discontinuities = np.zeros(max_out * 2, dtype=np.int32)

    if _lib is not None:
        count = _lib.wu_mv_scan_horizontal(
            mv_row.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            width_mbs,
            threshold_sq,
            discontinuities.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            max_out
        )
        return [(int(discontinuities[i*2]), int(discontinuities[i*2+1])) for i in range(count)]

    # Fallback
    result = []
    for i in range(width_mbs - 1):
        dx = int(mv_row[i * 2]) - int(mv_row[(i + 1) * 2])
        dy = int(mv_row[i * 2 + 1]) - int(mv_row[(i + 1) * 2 + 1])
        mag_sq = dx * dx + dy * dy
        if mag_sq >= threshold_sq and len(result) < max_out:
            result.append((i, mag_sq))
    return result


def mv_coherence_score(mv_field: 'np.ndarray', width_mbs: int, height_mbs: int) -> int:
    """
    Compute MV field coherence score.

    Args:
        mv_field: All MVs as [mvx, mvy] pairs (row-major, int16)
        width_mbs: Frame width in macroblocks
        height_mbs: Frame height in macroblocks

    Returns:
        Coherence score (0-100, higher = more coherent)
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    mv_field = np.ascontiguousarray(mv_field.ravel(), dtype=np.int16)

    if _lib is not None:
        return _lib.wu_mv_coherence_score(
            mv_field.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            width_mbs,
            height_mbs
        )

    # Simplified fallback
    if width_mbs < 3 or height_mbs < 3:
        return 100

    total_gradient = 0
    count = 0
    for y in range(1, height_mbs - 1):
        for x in range(1, width_mbs - 1):
            idx = (y * width_mbs + x) * 2
            mvx, mvy = int(mv_field[idx]), int(mv_field[idx + 1])
            # Check left neighbour
            dx = mvx - int(mv_field[idx - 2])
            dy = mvy - int(mv_field[idx - 1])
            total_gradient += dx * dx + dy * dy
            count += 1

    if count == 0:
        return 100
    avg = total_gradient // count
    normalised = min(avg // 100, 100)
    return 100 - normalised


def mv_detect_outliers(mv_field: 'np.ndarray', width_mbs: int, height_mbs: int,
                       threshold: int = 2500, max_out: int = 100):
    """
    Detect MV outliers that deviate from local neighbourhood.

    Args:
        mv_field: All MVs as [mvx, mvy] pairs (row-major, int16)
        width_mbs: Frame width in macroblocks
        height_mbs: Frame height in macroblocks
        threshold: Deviation squared threshold (e.g. 2500 = 50^2)
        max_out: Maximum outliers to record

    Returns:
        List of (mb_x, mb_y, deviation_sq) tuples
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    mv_field = np.ascontiguousarray(mv_field.ravel(), dtype=np.int16)
    outliers = np.zeros(max_out * 3, dtype=np.int32)

    if _lib is not None:
        count = _lib.wu_mv_detect_outliers(
            mv_field.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            width_mbs,
            height_mbs,
            outliers.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            threshold,
            max_out
        )
        return [(int(outliers[i*3]), int(outliers[i*3+1]), int(outliers[i*3+2]))
                for i in range(count)]

    # Fallback returns empty for simplicity
    return []


def scan_nal_units(buffer: bytes, max_nals: int = 1000):
    """
    Scan buffer for NAL start codes and extract NAL type sequence.

    Args:
        buffer: H.264 bitstream data
        max_nals: Maximum NALs to record

    Returns:
        List of (nal_type, byte_offset) tuples
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    buf_arr = np.frombuffer(buffer, dtype=np.uint8)
    nal_types = np.zeros(max_nals, dtype=np.uint8)
    nal_offsets = np.zeros(max_nals, dtype=np.int32)

    if _lib is not None:
        count = _lib.wu_scan_nal_units(
            buf_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(buf_arr),
            nal_types.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            nal_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            max_nals
        )
        return [(int(nal_types[i]), int(nal_offsets[i])) for i in range(count)]

    # Fallback
    result = []
    i = 0
    while i < len(buf_arr) - 4 and len(result) < max_nals:
        if buf_arr[i] == 0 and buf_arr[i + 1] == 0:
            if buf_arr[i + 2] == 1:
                nal_type = buf_arr[i + 3] & 0x1F
                result.append((int(nal_type), i))
                i += 3
            elif buf_arr[i + 2] == 0 and buf_arr[i + 3] == 1 and i + 4 < len(buf_arr):
                nal_type = buf_arr[i + 4] & 0x1F
                result.append((int(nal_type), i))
                i += 4
            else:
                i += 1
        else:
            i += 1
    return result


def analyse_nal_sequence(nal_types, max_anomalies: int = 100):
    """
    Analyse NAL type sequence for structural anomalies.

    Anomaly codes:
        1 = Missing SPS before first slice
        2 = Missing PPS before first slice
        3 = SPS after slice (potential splice)
        4 = PPS after slice (potential splice)
        5 = Non-IDR slice without preceding IDR
        6 = Invalid NAL type (>31)
        7 = Consecutive IDRs (unusual)

    Args:
        nal_types: Sequence of NAL types (list or array)
        max_anomalies: Maximum anomalies to record

    Returns:
        List of (index, anomaly_code) tuples
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    nal_arr = np.ascontiguousarray(nal_types, dtype=np.uint8)
    anomalies = np.zeros(max_anomalies * 2, dtype=np.int32)

    if _lib is not None:
        count = _lib.wu_analyse_nal_sequence(
            nal_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(nal_arr),
            anomalies.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            max_anomalies
        )
        return [(int(anomalies[i*2]), int(anomalies[i*2+1])) for i in range(count)]

    # Simplified fallback
    return []


def count_epb(buffer: bytes) -> int:
    """
    Count emulation prevention bytes (00 00 03 sequences).

    Args:
        buffer: NAL data

    Returns:
        Number of EPB sequences found
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    buf_arr = np.frombuffer(buffer, dtype=np.uint8)

    if _lib is not None:
        return _lib.wu_count_epb(
            buf_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(buf_arr)
        )

    # Fallback
    count = 0
    i = 0
    while i < len(buf_arr) - 2:
        if buf_arr[i] == 0 and buf_arr[i + 1] == 0 and buf_arr[i + 2] == 3:
            count += 1
            i += 3
        else:
            i += 1
    return count


def entropy_stats(buffer: bytes):
    """
    Compute entropy statistics on slice data.

    Args:
        buffer: Slice data after header

    Returns:
        Dict with keys: zero_count, one_count, byte_sum, entropy_est
    """
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")

    buf_arr = np.frombuffer(buffer, dtype=np.uint8)
    stats = np.zeros(4, dtype=np.int32)

    if _lib is not None:
        _lib.wu_entropy_stats(
            buf_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(buf_arr),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        )
    else:
        # Fallback
        zero_count = 0
        one_count = 0
        byte_sum = 0
        for b in buf_arr:
            byte_sum += int(b)
            ones = bin(b).count('1')
            one_count += ones
            zero_count += 8 - ones
        stats[0] = zero_count
        stats[1] = one_count
        stats[2] = byte_sum
        stats[3] = 50  # Default estimate

    return {
        'zero_count': int(stats[0]),
        'one_count': int(stats[1]),
        'byte_sum': int(stats[2]),
        'entropy_est': int(stats[3])
    }
