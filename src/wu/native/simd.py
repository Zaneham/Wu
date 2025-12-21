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

