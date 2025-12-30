# Wu Forensics Performance Optimization Plan

## Executive Summary

This document outlines a strategic plan for optimizing Wu Forensics for large-scale video processing. A 5-minute video at 30fps contains ~9,000 frames, and current Python/NumPy implementations become bottlenecks at this scale. The plan prioritizes C extensions and targeted assembly optimizations.

**Dual Purpose:** Beyond performance, this plan also addresses IP protection through strategic use of assembly language. Critical forensic algorithms implemented in assembly provide both maximum performance and a layer of protection against direct code copying, while remaining open source at the API level.

## Current State Assessment

### Existing Native Infrastructure
✅ **Already Implemented:**
- Native SIMD library (`wu_simd.c`) with AVX2/SSE2/NEON support
- Core functions: dot product, euclidean distance, normalize, variance
- Sobel 3x3 gradient computation
- DCT 8x8 computation
- Blockiness computation
- Python bindings via ctypes

### Performance Bottlenecks Identified

**Critical (10-100x speedup potential):**
1. **Copy-Move Detection** (`copymove.py`)
   - DCT computation for thousands of blocks
   - O(n²) similarity search (all-pairs comparison)
   - Block extraction and normalization

2. **Block Grid Analysis** (`blockgrid.py`)
   - Blockiness computation (64 offset combinations per region)
   - Batch DCT for double compression detection
   - Regional grid offset detection

3. **Lighting Analysis** (`lighting.py`)
   - Sobel gradient computation (already native, but can be improved)
   - Regional lighting estimation (hundreds of regions)
   - Weighted statistics computation

4. **PRNU Analysis** (`prnu.py`)
   - Wiener filtering (expensive convolution)
   - FFT-based cross-correlation
   - Noise residual extraction

**High Priority (5-20x speedup potential):**
5. **Visual Analysis** (`visual.py`)
   - ELA: JPEG resave + pixel difference
   - Quantization table analysis

6. **Video Frame Processing**
   - Frame extraction and conversion
   - Batch processing pipeline

---

## Optimization Strategy

### Phase 1: C Extensions (Immediate Impact, Lower Risk)

**Priority: HIGH** - These provide significant speedups with manageable complexity.

#### 1.1 Copy-Move Detection (`copymove_native.c`)

**Functions to implement:**
```c
// Batch DCT feature extraction
int wu_extract_dct_features_batch(
    const float* image,           // Input grayscale image
    int width, int height,
    int block_size,               // Typically 16
    int step,                     // Step between blocks
    int n_coefficients,           // Features per block (16)
    float* features,              // Output: [n_blocks, n_coefficients]
    int* positions,               // Output: [n_blocks, 2] (x, y)
    int max_blocks
);

// Fast similarity search with spatial filtering
int wu_find_similar_blocks_optimized(
    const float* features,        // Normalized feature vectors
    int n_blocks,
    int n_features,
    float threshold,
    float min_distance,
    const int* positions,
    float* matches,               // Output: [n_matches, 3] (i, j, similarity)
    int max_matches
);
```

**Optimization techniques:**
- **OpenMP parallelization**: Process blocks in parallel
- **SIMD DCT**: Use AVX2 for 8x8 DCT (already have 8x8, extend to 16x16)
- **Spatial indexing**: Use grid-based spatial hash to reduce comparisons
- **Early termination**: Skip blocks with insufficient variance
- **Memory alignment**: 32-byte aligned allocations for AVX2

**Expected speedup:** 15-30x for block extraction, 10-20x for similarity search

#### 1.2 Block Grid Analysis (`blockgrid_native.c`)

**Functions to implement:**
```c
// Parallel blockiness computation for all offsets
void wu_compute_blockiness_all_offsets(
    const double* image,
    int width, int height,
    int block_size,
    double* scores              // Output: [8, 8] scores for each offset
);

// Batch DCT coefficient extraction for double compression
void wu_extract_dct_coefficients_batch(
    const double* image,
    int width, int height,
    int block_size,
    int max_samples,
    double* coefficients,       // Output: flattened non-DC coefficients
    int* n_coefficients          // Output: actual count
);
```

**Optimization techniques:**
- **SIMD blockiness**: Process 4 boundary pixels at once with AVX2
- **Parallel offset testing**: Test all 64 offset combinations in parallel
- **Batch DCT**: Use FFTW for DCT (much faster than scipy)
- **Cache blocking**: Process image in tiles for better cache locality

**Expected speedup:** 20-40x for blockiness, 10-15x for DCT extraction

#### 1.3 Lighting Analysis (`lighting_native.c`)

**Functions to implement:**
```c
// Parallel regional lighting estimation
void wu_estimate_lighting_regions(
    const double* gray,
    const double* gx,
    const double* gy,
    int width, int height,
    int region_size,
    int step,
    double* light_vectors,        // Output: [n_regions, 3] (azimuth, elevation, confidence)
    int* n_regions
);

// Fast weighted statistics
void wu_compute_weighted_gradient_stats(
    const double* gx,
    const double* gy,
    const double* weights,
    int n_pixels,
    double* weighted_gx,         // Output
    double* weighted_gy,          // Output
    double* variance              // Output
);
```

**Optimization techniques:**
- **SIMD weighted sums**: Use AVX2 FMA for weighted accumulation
- **Parallel region processing**: OpenMP for independent regions
- **Separable Sobel**: Already optimized, but can improve cache usage

**Expected speedup:** 10-15x for regional analysis

#### 1.4 PRNU Analysis (`prnu_native.c`)

**Functions to implement:**
```c
// Fast Wiener denoising using integral images
void wu_wiener_denoise_fast(
    const double* input,
    double* output,
    int width, int height,
    int window_size
);

// FFTW-based cross-correlation
void wu_compute_pce_fftw(
    const double* noise,
    const double* fingerprint,
    int width, int height,
    double* pce,                 // Output: Peak-to-Correlation Energy
    double* p_value              // Output: Statistical significance
);
```

**Optimization techniques:**
- **Integral images**: O(1) local mean/variance lookup
- **FFTW**: Much faster than numpy FFT (3-5x)
- **SIMD element-wise ops**: AVX2 for noise subtraction
- **Parallel channel processing**: Process RGB channels in parallel

**Expected speedup:** 20-50x for Wiener filter, 5-10x for FFT

---

### Phase 2: Assembly Optimizations (Maximum Performance + IP Protection)

**Priority: HIGH** - Critical algorithms benefit from both performance and protection.

#### 2.1 Strategic Assembly Usage: Performance + IP Protection

**Dual Benefits:**
1. **Performance**: Hand-optimized assembly can outperform compiler-generated code by 10-30%
2. **IP Protection**: Assembly code is significantly harder to reverse-engineer than C, providing a layer of protection for proprietary algorithms

**Use Assembly for:**
- **Core forensic algorithms** (your competitive advantage)
  - Copy-move detection similarity matching
  - PRNU cross-correlation computation
  - Block grid offset detection
  - Lighting direction estimation
- Extremely hot loops called millions of times
- Operations not well-optimized by compilers
- Custom instruction sequences (e.g., horizontal reductions)
- Platform-specific optimizations (e.g., AVX-512 on newer CPUs)

**Use C with Intrinsics for:**
- Utility functions (less valuable IP)
- Cross-platform portability (where needed)
- Easier maintenance and debugging (for non-critical paths)

**Recommendation:** Implement critical forensic algorithms directly in assembly from the start. This provides both maximum performance and IP protection without the overhead of porting later.

#### 2.2 IP Protection Strategy

**What to Protect (High Value Algorithms):**
1. **Copy-Move Detection Core**
   - DCT-based feature extraction with custom coefficients
   - Similarity matching algorithm (your specific thresholds/heuristics)
   - Spatial filtering logic

2. **PRNU Analysis**
   - Wiener filter implementation details
   - Cross-correlation computation
   - PCE (Peak-to-Correlation Energy) calculation

3. **Block Grid Analysis**
   - Blockiness computation algorithm
   - Grid offset detection heuristics
   - Double compression detection logic

4. **Lighting Analysis**
   - Light direction estimation from gradients
   - Regional consistency checking
   - Specular highlight detection

**Protection Techniques:**

1. **Obfuscated Assembly**
   - Use non-obvious register allocation
   - Insert dummy operations that don't affect results
   - Use equivalent but different instruction sequences
   - Mix data and code (carefully, for security)

2. **Platform-Specific Implementations**
   - x86-64 (AVX2) - most common
   - ARM64 (NEON) - for Apple Silicon, servers
   - Keep source separate per platform (harder to compare)

3. **Function-Level Protection**
   - Critical functions: Pure assembly
   - Supporting functions: C with intrinsics (less valuable)
   - API layer: Python/C (public interface)

4. **Build-Time Obfuscation**
   - Compile with `-O3 -fno-ident` (remove debug symbols)
   - Strip symbols: `strip -s`
   - Use custom calling conventions (platform-specific)

**Example Protection Pattern:**
```asm
; Copy-move similarity matching (obfuscated)
; This implements a proprietary similarity metric
wu_similarity_match_avx2:
    ; Non-obvious register usage
    mov r10, rdi          ; features_a (disguised)
    mov r11, rsi          ; features_b (disguised)
    
    ; Dummy operations that don't affect result
    vxorps ymm15, ymm15, ymm15  ; Clear (looks like init, but unused)
    
    ; Actual computation with obfuscated flow
    vmovaps ymm0, [r10]   ; Load first 8 features
    vmovaps ymm1, [r11]
    vsubps ymm2, ymm0, ymm1     ; Difference
    vmulps ymm2, ymm2, ymm2     ; Square
    ; ... continue with proprietary algorithm
    
    ; Insert dummy operations periodically
    vaddps ymm15, ymm15, ymm15  ; No-op (maintains appearance)
    
    ret
```

**Legal Considerations:**
- ✅ **Open Source API**: Python interface remains open and documented
- ✅ **Binary Distribution**: Assembly code can be distributed as compiled binaries
- ✅ **Algorithm Protection**: Your specific implementation details are protected
- ⚠️ **License Clarity**: Consider adding to license: "Core algorithms implemented in assembly for performance. Source available for review upon request for security auditing."
- ⚠️ **Patent Issues**: Assembly doesn't protect against patent claims, only copyright

**Maintenance Strategy:**
- Keep commented assembly source in private repo
- Public repo contains Python API + binary libraries
- Document algorithm at high level (what it does, not how)
- Version control: Tag assembly versions separately

#### 2.3 Target Functions for Assembly Implementation

**Priority 1: Core Forensic Algorithms (High IP Value)**

**1. Copy-Move Similarity Matching (Proprietary Algorithm)**
```asm
; wu_similarity_match_proprietary_avx2
; Implements proprietary similarity metric with custom thresholds
; This is your competitive advantage - protect it
.global wu_similarity_match_proprietary_avx2
wu_similarity_match_proprietary_avx2:
    ; Input: RDI = features_a, RSI = features_b, RDX = n_features
    ; Output: XMM0 = similarity score (0.0-1.0)
    
    ; Obfuscated register usage
    mov r10, rdi          ; features_a
    mov r11, rsi          ; features_b
    mov rcx, rdx          ; n_features
    
    vxorps ymm7, ymm7, ymm7  ; Accumulator for dot product
    vxorps ymm8, ymm8, ymm8  ; Accumulator for norms
    
    ; Process 8 features at a time (AVX2 width)
    .loop:
        vmovaps ymm0, [r10]      ; Load 8 features from A
        vmovaps ymm1, [r11]      ; Load 8 features from B
        
        ; Proprietary similarity computation
        vmulps ymm2, ymm0, ymm1  ; Element-wise product
        vaddps ymm7, ymm7, ymm2  ; Accumulate dot product
        
        ; Compute norms (part of proprietary metric)
        vmulps ymm3, ymm0, ymm0  ; A squared
        vmulps ymm4, ymm1, ymm1  ; B squared
        vaddps ymm5, ymm3, ymm4  ; Combined
        vaddps ymm8, ymm8, ymm5  ; Accumulate
        
        add r10, 32              ; Next 8 floats
        add r11, 32
        sub rcx, 8
        jg .loop
    
    ; Horizontal reduction (obfuscated)
    vperm2f128 ymm9, ymm7, ymm7, 0x01
    vaddps ymm7, ymm7, ymm9
    vhaddps ymm7, ymm7, ymm7
    vhaddps ymm7, ymm7, ymm7
    
    ; Final proprietary computation (your secret sauce)
    ; This is where your algorithm differs from standard cosine similarity
    vmovss xmm0, xmm7
    ret
```

**2. PRNU Cross-Correlation (Critical IP)**
```asm
; wu_prnu_cross_correlation_avx2
; Proprietary PCE computation with custom normalization
.global wu_prnu_cross_correlation_avx2
wu_prnu_cross_correlation_avx2:
    ; Input: RDI = noise, RSI = fingerprint, RDX = width, RCX = height
    ; Output: XMM0 = PCE value, XMM1 = p-value
    
    ; Your specific cross-correlation algorithm
    ; Obfuscated with dummy operations and non-obvious flow
    ; ... implementation details protected ...
    ret
```

**3. Block Grid Offset Detection (Proprietary Heuristics)**
```asm
; wu_detect_grid_offset_proprietary_avx2
; Implements your specific blockiness scoring algorithm
.global wu_detect_grid_offset_proprietary_avx2
wu_detect_grid_offset_proprietary_avx2:
    ; Input: RDI = image, RDX = width, RCX = height
    ; Output: RAX = best_x_offset, RDX = best_y_offset
    
    ; Test all 64 offset combinations with proprietary scoring
    ; Your specific heuristics for confidence calculation
    ; ... protected implementation ...
    ret
```

**Priority 2: Performance-Critical Utilities (Lower IP Value)**

**Horizontal Reduction (Utility - Less Critical)**
```asm
; Horizontal sum of 8 floats in YMM register
; Input: YMM0 = [a0, a1, a2, a3, a4, a5, a6, a7]
; Output: XMM0 = sum
horizontal_sum_avx2:
    vperm2f128 ymm1, ymm0, ymm0, 0x01  ; Swap halves
    vaddps ymm0, ymm0, ymm1             ; Add halves
    vhaddps ymm0, ymm0, ymm0            ; Horizontal add
    vhaddps ymm0, ymm0, ymm0            ; Final horizontal add
    ret
```

**Boundary Difference Computation (Utility)**
```asm
; Compute 4 boundary differences at once
compute_boundary_diff_avx2:
    vmovapd ymm0, [rdi + rcx*8]      ; Column x
    vmovapd ymm1, [rdi + (rcx+1)*8]  ; Column x+1
    vsubpd ymm2, ymm0, ymm1          ; Difference
    vmulpd ymm2, ymm2, ymm2          ; Square
    vaddpd ymm3, ymm3, ymm2          ; Accumulate
    ret
```

**DCT Computation Strategy:**
- **For proprietary DCT variants**: Implement in assembly
- **For standard DCT**: Use FFTW (well-optimized, not your IP)
- **Hybrid approach**: Use FFTW for standard, assembly for your custom coefficients

---

### Phase 3: Video Processing Pipeline

**Priority: HIGH** - Critical for the stated use case.

#### 3.1 Frame Extraction Optimization

**Current bottleneck:** PIL/OpenCV frame extraction is slow for thousands of frames.

**Solution:**
```c
// Fast frame extraction using FFmpeg directly
typedef struct {
    uint8_t* data;          // RGB24 frame data
    int width, height;
    int64_t timestamp;      // Frame timestamp in microseconds
} Frame;

int wu_extract_frames_fast(
    const char* video_path,
    Frame* frames,          // Pre-allocated buffer
    int max_frames,
    double fps,             // Target FPS (for subsampling)
    int* n_frames           // Output: actual frames extracted
);
```

**Optimization techniques:**
- **FFmpeg hardware decoding**: Use GPU or hardware decoder when available
- **Frame subsampling**: Don't process every frame (e.g., every 10th frame)
- **Parallel extraction**: Extract frames in parallel chunks
- **Memory-mapped I/O**: Reduce copy overhead

**Expected speedup:** 5-10x for frame extraction

#### 3.2 Batch Frame Processing

**Architecture:**
```c
// Process multiple frames in parallel
typedef struct {
    const Frame* frames;
    int n_frames;
    AnalysisConfig* config;
    DimensionResult* results;  // Output: one per frame
} BatchJob;

void wu_process_frames_batch(
    BatchJob* job,
    int n_threads
);
```

**Optimization techniques:**
- **Thread pool**: Reuse threads instead of spawning
- **Work stealing**: Dynamic load balancing
- **SIMD batch ops**: Process multiple frames' features simultaneously
- **Cache-friendly access**: Process frames in spatial order

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. ✅ Extend `wu_simd.c` with batch operations
2. ✅ Add OpenMP support to build system
3. ✅ Create `copymove_native.c` with DCT batch extraction
4. ✅ Benchmark against Python implementation

### Phase 2: Core Algorithms (Weeks 3-4)
1. ✅ Implement `blockgrid_native.c` with parallel blockiness
2. ✅ Implement `lighting_native.c` with regional analysis
3. ✅ Add FFTW dependency for PRNU
4. ✅ Implement `prnu_native.c` with Wiener filter

### Phase 3: Video Pipeline (Weeks 5-6)
1. ✅ Add FFmpeg integration for frame extraction
2. ✅ Implement batch frame processing
3. ✅ Add frame subsampling/skipping logic
4. ✅ Benchmark on 5-minute video

### Phase 4: Assembly Implementation (Weeks 7-10)
1. ✅ Implement core algorithms in assembly (copy-move, PRNU, block grid)
2. ✅ Add obfuscation techniques for IP protection
3. ✅ Create platform-specific builds (x86-64, ARM64)
4. ✅ Profile and optimize assembly code
5. ✅ Validate correctness against C reference
6. ✅ Set up binary distribution pipeline

---

## Architecture Plans

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Application Layer                  │
│  (wu/analyzer.py, wu/dimensions/*.py)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ ctypes calls
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    C Wrapper Layer                           │
│  (c_wrappers/*.c)                                            │
│  - Input validation                                          │
│  - Memory alignment checks                                   │
│  - Error handling                                            │
│  - Fallback to C implementation if needed                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Function calls
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Assembly Implementation Layer                   │
│  (assembly/x86_64/*.asm, assembly/arm64/*.asm)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Copy-Move    │  │ PRNU         │  │ Block Grid   │     │
│  │ Similarity   │  │ Correlation  │  │ Detection    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Lighting     │  │ Common Utils  │                        │
│  │ Estimation   │  │ (horizontal  │                        │
│  │              │  │  reductions) │                        │
│  └──────────────┘  └──────────────┘                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Uses
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              C Utility Layer (Existing)                     │
│  (wu_simd.c)                                                │
│  - Dot product, normalize, variance                         │
│  - Sobel gradients                                          │
│  - DCT 8x8                                                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Image/Video Input
    │
    ▼
┌─────────────────┐
│ Frame Extraction │ (FFmpeg - C)
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Grayscale Conv  │ (C with SIMD)
└────────┬─────────┘
         │
         ├─────────────────┬─────────────────┬──────────────┐
         ▼                 ▼                 ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐
│ Copy-Move    │  │ Block Grid   │  │ Lighting     │  │ PRNU     │
│ Detection    │  │ Analysis     │  │ Analysis     │  │ Analysis │
│              │  │              │  │              │  │          │
│ 1. Extract   │  │ 1. Compute   │  │ 1. Compute   │  │ 1. Wiener│
│    blocks    │  │    blockiness│  │    gradients │  │    filter│
│              │  │              │  │              │  │          │
│ 2. DCT       │  │ 2. Test 64   │  │ 2. Estimate  │  │ 2. FFT   │
│    features  │  │    offsets   │  │    direction │  │          │
│              │  │              │  │              │  │          │
│ 3. Similarity│  │ 3. Find best │  │ 3. Regional  │  │ 3. Cross │
│    matching  │  │    offset    │  │    analysis  │  │    corr  │
│    (ASM)     │  │    (ASM)     │  │    (ASM)     │  │    (ASM) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────┬────┘
       │                 │                 │              │
       └─────────────────┴─────────────────┴──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   Results    │
                   │  Aggregation │
                   └──────────────┘
```

### Assembly Module Architecture

#### Directory Structure
```
src/wu/native/
├── wu_simd.c              # Existing C with intrinsics (utilities)
├── wu_simd.h              # Header file
├── wu_simd.dll/.so        # Compiled library
│
├── assembly/              # NEW: Assembly implementations
│   ├── x86_64/           # x86-64 (Intel/AMD) implementations
│   │   ├── copymove.asm  # Copy-move similarity matching
│   │   ├── prnu.asm      # PRNU cross-correlation
│   │   ├── blockgrid.asm # Block grid offset detection
│   │   ├── lighting.asm  # Lighting direction estimation
│   │   └── common.asm    # Shared utilities (horizontal sums, etc.)
│   │
│   ├── arm64/            # ARM64 (NEON) implementations
│   │   ├── copymove.asm  # ARM64 version
│   │   ├── prnu.asm      # ARM64 version
│   │   ├── blockgrid.asm # ARM64 version
│   │   ├── lighting.asm  # ARM64 version
│   │   └── common.asm    # ARM64 shared utilities
│   │
│   └── Makefile          # Build script for assembly files
│
├── c_wrappers/            # NEW: C wrappers for assembly functions
│   ├── copymove_wrapper.c
│   ├── prnu_wrapper.c
│   ├── blockgrid_wrapper.c
│   └── lighting_wrapper.c
│
└── build.py               # Updated build script
```

#### Build Pipeline

**Step 1: Assemble .asm → .o**
```bash
# x86-64 (YASM/NASM)
yasm -f win64 -o copymove_x86_64.o copymove.asm    # Windows
yasm -f elf64 -o copymove_x86_64.o copymove.asm    # Linux
yasm -f macho64 -o copymove_x86_64.o copymove.asm  # macOS

# ARM64 (GAS)
as -64 -o copymove_arm64.o copymove.asm
```

**Step 2: Compile C wrappers**
```bash
gcc -c -O3 -mavx2 -fPIC copymove_wrapper.c -o copymove_wrapper.o
```

**Step 3: Link into shared library**
```bash
# Windows
gcc -shared -o wu_core.dll copymove_x86_64.o copymove_wrapper.o wu_simd.o

# Linux
gcc -shared -o libwu_core.so copymove_x86_64.o copymove_wrapper.o wu_simd.o

# macOS
gcc -shared -o libwu_core.dylib copymove_x86_64.o copymove_wrapper.o wu_simd.o
```

**Step 4: Strip symbols (IP protection)**
```bash
strip -s wu_core.dll/libwu_core.so/libwu_core.dylib
```

---

### Function Architecture: Copy-Move Similarity Matching

#### C Interface (Public API)
```c
// copymove_wrapper.c
double wu_similarity_match_proprietary(
    const float* features_a,
    const float* features_b,
    size_t n_features
);
```

#### Assembly Implementation Structure

**x86-64 (AVX2) Architecture:**
```asm
; copymove.asm (x86-64)
section .text
global wu_similarity_match_proprietary_avx2

wu_similarity_match_proprietary_avx2:
    ; Function prologue
    push rbp
    mov rbp, rsp
    
    ; Input: RDI = features_a, RSI = features_b, RDX = n_features
    ; Save non-volatile registers
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Register allocation (obfuscated)
    mov r10, rdi          ; features_a (via intermediate)
    mov r11, rsi          ; features_b (via intermediate)
    mov rcx, rdx          ; n_features
    
    ; Initialize accumulators
    vxorps ymm7, ymm7, ymm7  ; Dot product accumulator
    vxorps ymm8, ymm8, ymm8  ; Norm accumulator (proprietary)
    
    ; Dummy initialization (obfuscation)
    vxorps ymm15, ymm15, ymm15  ; Unused, but looks important
    
    ; Main loop: Process 8 floats at a time
    .loop:
        ; Load 8 features from each array
        vmovaps ymm0, [r10]      ; features_a[i:i+8]
        vmovaps ymm1, [r11]      ; features_b[i:i+8]
        
        ; Proprietary similarity computation
        vmulps ymm2, ymm0, ymm1  ; Element-wise product
        vfmadd231ps ymm7, ymm2, ymm2  ; Accumulate (FMA for performance)
        
        ; Compute norms (part of proprietary metric)
        vmulps ymm3, ymm0, ymm0  ; a²
        vmulps ymm4, ymm1, ymm1  ; b²
        vaddps ymm5, ymm3, ymm4  ; Combined norm
        vfmadd231ps ymm8, ymm5, ymm5  ; Accumulate norm (proprietary)
        
        ; Dummy operation (obfuscation)
        vaddps ymm15, ymm15, ymm15  ; No-op
        
        ; Advance pointers
        add r10, 32              ; Next 8 floats (8 * 4 bytes)
        add r11, 32
        sub rcx, 8
        jg .loop
    
    ; Handle remainder (scalar)
    .remainder:
        test rcx, rcx
        jz .reduce
        
        ; Process remaining 1-7 elements
        movss xmm0, [r10]
        movss xmm1, [r11]
        mulss xmm0, xmm1
        addss xmm7, xmm0
        
        add r10, 4
        add r11, 4
        dec rcx
        jnz .remainder
    
    ; Horizontal reduction (obfuscated sequence)
    .reduce:
        ; Reduce ymm7 (8 floats) to single float
        vperm2f128 ymm9, ymm7, ymm7, 0x01  ; Swap halves
        vaddps ymm7, ymm7, ymm9             ; Add halves
        vhaddps ymm7, ymm7, ymm7           ; Horizontal add
        vhaddps ymm7, ymm7, ymm7           ; Final horizontal add
        vmovss xmm0, xmm7                  ; Extract result
        
        ; Reduce ymm8 similarly
        vperm2f128 ymm10, ymm8, ymm8, 0x01
        vaddps ymm8, ymm8, ymm10
        vhaddps ymm8, ymm8, ymm8
        vhaddps ymm8, ymm8, ymm8
        vmovss xmm1, xmm8
    
    ; Final proprietary computation (secret sauce)
    ; This is where your algorithm differs from standard cosine similarity
    vdivss xmm0, xmm0, xmm1    ; Proprietary normalization
    vmovss xmm0, xmm0, [rel .proprietary_scale]  ; Apply proprietary scale
    vmulss xmm0, xmm0, xmm0    ; Square (proprietary transformation)
    
    ; Function epilogue
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

section .data
.proprietary_scale: dd 1.41421356  ; Obfuscated constant (sqrt(2))
.dummy_constant: dd 0.12345678     ; Dummy (obfuscation)
```

**ARM64 (NEON) Architecture:**
```asm
; copymove.asm (ARM64)
.section .text
.global wu_similarity_match_proprietary_neon

wu_similarity_match_proprietary_neon:
    ; Function prologue
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    
    ; Input: X0 = features_a, X1 = features_b, X2 = n_features
    ; Save non-volatile registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    
    ; Register allocation
    mov x19, x0          ; features_a
    mov x20, x1          ; features_b
    mov x21, x2          ; n_features
    
    ; Initialize accumulators (NEON)
    movi v7.4s, #0      ; Dot product accumulator
    movi v8.4s, #0      ; Norm accumulator
    
    ; Main loop: Process 4 floats at a time (NEON width)
    .loop:
        ; Load 4 features from each array
        ldr q0, [x19], #16   ; features_a[i:i+4]
        ldr q1, [x20], #16   ; features_b[i:i+4]
        
        ; Proprietary similarity computation
        fmul v2.4s, v0.4s, v1.4s    ; Element-wise product
        fmla v7.4s, v2.4s, v2.4s    ; Accumulate (FMA)
        
        ; Compute norms
        fmul v3.4s, v0.4s, v0.4s   ; a²
        fmul v4.4s, v1.4s, v1.4s   ; b²
        fadd v5.4s, v3.4s, v4.4s   ; Combined
        fmla v8.4s, v5.4s, v5.4s   ; Accumulate
        
        ; Advance
        subs x21, x21, #4
        bgt .loop
    
    ; Horizontal reduction
    faddp v7.4s, v7.4s, v7.4s   ; Pairwise add
    faddp v7.4s, v7.4s, v7.4s   ; Final add
    faddp s0, v7.2s             ; Extract scalar
    
    faddp v8.4s, v8.4s, v8.4s
    faddp v8.4s, v8.4s, v8.4s
    faddp s1, v8.2s
    
    ; Final proprietary computation
    fdiv s0, s0, s1
    ldr s2, .proprietary_scale
    fmul s0, s0, s2
    fmul s0, s0, s0
    
    ; Function epilogue
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

.section .data
.proprietary_scale: .float 1.41421356
```

---

### Detailed Implementation: Copy-Move Similarity (Optimized)

**Enhanced Implementation with Intel Optimization Techniques:**

```asm
; Optimized version with loop unrolling and prefetching
wu_similarity_match_proprietary_avx2_optimized:
    push rbp
    mov rbp, rsp
    
    ; Save registers
    push rbx
    push r12
    push r13
    
    ; Input: RDI = features_a, RSI = features_b, RDX = n_features
    mov r10, rdi
    mov r11, rsi
    mov rcx, rdx
    
    ; Initialize accumulators (use multiple for better ILP)
    vxorps ymm7, ymm7, ymm7  ; Dot product accumulator 1
    vxorps ymm8, ymm8, ymm8  ; Dot product accumulator 2
    vxorps ymm9, ymm9, ymm9  ; Norm accumulator 1
    vxorps ymm10, ymm10, ymm10 ; Norm accumulator 2
    
    ; Prefetch first cache lines
    prefetchnta [r10 + 0]
    prefetchnta [r11 + 0]
    prefetchnta [r10 + 64]
    prefetchnta [r11 + 64]
    
    ; Main loop: Process 32 floats per iteration (4 AVX2 registers)
    ; This reduces loop overhead and improves ILP
    .loop:
        ; Unroll 4x: Process 32 floats (4 * 8 floats)
        ; Iteration 1
        vmovaps ymm0, [r10 + 0]    ; Load 8 floats from A
        vmovaps ymm1, [r11 + 0]    ; Load 8 floats from B
        vfmadd231ps ymm7, ymm0, ymm1  ; Dot product (FMA: a*b + acc)
        vfmadd231ps ymm9, ymm0, ymm0  ; Norm A (a*a + acc)
        vfmadd231ps ymm9, ymm1, ymm1  ; Norm B (b*b + acc)
        
        ; Iteration 2
        vmovaps ymm2, [r10 + 32]
        vmovaps ymm3, [r11 + 32]
        vfmadd231ps ymm8, ymm2, ymm3
        vfmadd231ps ymm10, ymm2, ymm2
        vfmadd231ps ymm10, ymm3, ymm3
        
        ; Iteration 3
        vmovaps ymm4, [r10 + 64]
        vmovaps ymm5, [r11 + 64]
        vfmadd231ps ymm7, ymm4, ymm5
        vfmadd231ps ymm9, ymm4, ymm4
        vfmadd231ps ymm9, ymm5, ymm5
        
        ; Iteration 4
        vmovaps ymm6, [r10 + 96]
        vmovaps ymm11, [r11 + 96]
        vfmadd231ps ymm8, ymm6, ymm11
        vfmadd231ps ymm10, ymm6, ymm6
        vfmadd231ps ymm10, ymm11, ymm11
        
        ; Prefetch next iteration (256 bytes ahead)
        prefetchnta [r10 + 256]
        prefetchnta [r11 + 256]
        
        ; Advance pointers
        add r10, 128              ; 32 floats * 4 bytes
        add r11, 128
        sub rcx, 32
        
        ; Continue loop if more than 32 elements remain
        cmp rcx, 32
        jge .loop
    
    ; Combine accumulators
    vaddps ymm7, ymm7, ymm8       ; Combine dot product accumulators
    vaddps ymm9, ymm9, ymm10      ; Combine norm accumulators
    
    ; Handle remainder (1-31 elements)
    test rcx, rcx
    jz .reduce
    
    .remainder:
        ; Process remaining elements 8 at a time
        cmp rcx, 8
        jl .remainder_scalar
        
        vmovaps ymm0, [r10]
        vmovaps ymm1, [r11]
        vfmadd231ps ymm7, ymm0, ymm1
        vfmadd231ps ymm9, ymm0, ymm0
        vfmadd231ps ymm9, ymm1, ymm1
        
        add r10, 32
        add r11, 32
        sub rcx, 8
        jnz .remainder
    
    .remainder_scalar:
        ; Process remaining 1-7 elements with scalar operations
        test rcx, rcx
        jz .reduce
        
        movss xmm0, [r10]
        movss xmm1, [r11]
        mulss xmm0, xmm1
        addss xmm7, xmm0
        
        movss xmm0, [r10]
        mulss xmm0, xmm0
        addss xmm9, xmm0
        
        movss xmm0, [r11]
        mulss xmm0, xmm0
        addss xmm9, xmm0
        
        add r10, 4
        add r11, 4
        dec rcx
        jnz .remainder_scalar
    
    ; Horizontal reduction (optimized sequence)
    .reduce:
        ; Reduce ymm7 (8 floats) to single float
        ; Method: vperm2f128 + vaddps (faster than vhaddps)
        vperm2f128 ymm8, ymm7, ymm7, 0x01  ; Swap high/low 128-bit halves
        vaddps ymm7, ymm7, ymm8            ; Add halves (4 floats remain)
        
        ; Extract to XMM and reduce further
        vextractf128 xmm8, ymm7, 1          ; Extract high 128 bits
        addps xmm7, xmm8                    ; Add (2 floats remain)
        
        ; Final reduction to scalar
        movshdup xmm8, xmm7                 ; Duplicate high float
        addss xmm7, xmm8                    ; Add (1 float)
        
        ; Same for norm accumulator
        vperm2f128 ymm10, ymm9, ymm9, 0x01
        vaddps ymm9, ymm9, ymm10
        vextractf128 xmm10, ymm9, 1
        addps xmm9, xmm10
        movshdup xmm10, xmm9
        addss xmm9, xmm10
    
    ; Final proprietary computation
    ; Similarity = dot_product / sqrt(norm_a * norm_b)
    ; But with proprietary scaling factor
    vmovss xmm0, xmm7                       ; Dot product
    vmovss xmm1, xmm9                       ; Combined norm
    
    ; Compute sqrt(norm) using fast approximation
    ; Or use: vsqrtss xmm1, xmm1, xmm1 (slower but accurate)
    vsqrtss xmm1, xmm1, xmm1
    
    ; Divide and apply proprietary scale
    vdivss xmm0, xmm0, xmm1
    vmulss xmm0, xmm0, [rel .proprietary_scale]
    
    ; Cleanup
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

section .data
.proprietary_scale: dd 1.41421356
```

**Key Optimization Techniques Applied:**

1. **4x Loop Unrolling:**
   - Processes 32 floats per iteration instead of 8
   - Reduces branch misprediction penalty
   - Better instruction scheduling

2. **Prefetching:**
   - `prefetchnta` for non-temporal access pattern
   - Prefetch 256 bytes ahead (4 cache lines)
   - Reduces memory latency

3. **FMA Instructions:**
   - `vfmadd231ps`: Fused Multiply-Add (1 cycle vs 2 cycles)
   - Better throughput on Haswell+ CPUs (2 FMA units)

4. **Optimized Horizontal Reduction:**
   - `vperm2f128` + `vaddps` (2 cycles) vs `vhaddps` (3 cycles)
   - `movshdup` for final reduction (faster than `haddps`)

5. **Register Allocation:**
   - Use multiple accumulators (ymm7, ymm8) for better ILP
   - Allows CPU to execute multiple FMA operations in parallel

6. **Memory Alignment:**
   - Use `vmovaps` (aligned) when possible
   - Falls back to `vmovups` (unaligned) with performance penalty

---

### Function Architecture: PRNU Cross-Correlation

#### C Interface
```c
void wu_prnu_cross_correlation_avx2(
    const double* noise,
    const double* fingerprint,
    int width,
    int height,
    double* pce,        // Output: Peak-to-Correlation Energy
    double* p_value     // Output: Statistical significance
);
```

#### Assembly Implementation Structure

**Key Operations:**
1. **FFT Preparation**: Convert real arrays to complex format
2. **FFT Computation**: Use FFTW (C library) for FFT, assembly for element-wise ops
3. **Cross-Correlation**: Multiply FFT results element-wise
4. **IFFT**: Inverse FFT (FFTW)
5. **Peak Detection**: Find maximum and compute PCE (assembly)

**Hybrid Approach:**
- FFT/IFFT: Use FFTW (proven, optimized)
- Element-wise operations: Assembly (SIMD)
- Peak detection: Assembly (proprietary algorithm)

**Detailed Implementation:**

```asm
; prnu.asm - Peak detection and PCE computation
section .text
global wu_compute_pce_avx2

wu_compute_pce_avx2:
    ; Input: RDI = correlation_result (from FFTW IFFT, double precision)
    ;        RSI = width, RDX = height
    ; Output: XMM0 = PCE, XMM1 = p-value
    
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    
    ; Calculate total elements
    mov rax, rsi
    imul rax, rdx          ; width * height
    mov rcx, rax           ; Loop counter
    
    ; Initialize peak tracking (use AVX2 for doubles: 4 at a time)
    vmovapd ymm0, [rdi]     ; Load first 4 doubles
    vmovapd ymm1, ymm0      ; Current maximum values
    vmovdqa ymm2, ymmword [rel .indices_init] ; Position indices (0,1,2,3)
    vmovdqa ymm3, ymm2      ; Maximum positions
    
    ; Initialize sum for PCE computation
    vxorpd ymm4, ymm4, ymm4 ; Sum accumulator
    vxorpd ymm5, ymm5, ymm5 ; Sum of squares accumulator
    
    ; Prefetch
    prefetchnta [rdi + 256]
    
    ; Main loop: Process 4 doubles at a time
    mov r8, rdi             ; Current pointer
    mov r9, 0               ; Global index counter
    add r8, 32              ; Skip first 4 (already loaded)
    sub rcx, 4
    
    .loop:
        ; Compare current values with maximum
        vmovapd ymm6, [r8]  ; Load next 4 doubles
        
        ; Find maximum values and their positions
        vcmppd ymm7, ymm6, ymm1, 1  ; Compare: ymm6 > ymm1 (CMPGT)
        vblendvpd ymm1, ymm1, ymm6, ymm7  ; Select maximum
        vblendvpd ymm3, ymm3, ymm2, ymm7  ; Update positions
        
        ; Accumulate for PCE computation
        vaddpd ymm4, ymm4, ymm6     ; Sum
        vfmadd231pd ymm5, ymm6, ymm6 ; Sum of squares (FMA)
        
        ; Update position indices
        vpaddq ymm2, ymm2, ymmword [rel .indices_step] ; Add 4 to each
        
        ; Prefetch next iteration
        prefetchnta [r8 + 256]
        
        ; Advance
        add r8, 32          ; Next 4 doubles
        add r9, 4
        sub rcx, 4
        jg .loop
    
    ; Handle remainder (1-3 elements)
    test rcx, rcx
    jz .find_global_max
    
    .remainder:
        movsd xmm6, [r8]
        movsd xmm7, xmm1    ; Extract first element of max
        comisd xmm6, xmm7
        jbe .no_update
        movsd xmm1, xmm6
        mov r10, r9
        movq xmm3, r10      ; Update position
    .no_update:
        addsd xmm4, xmm6
        mulsd xmm6, xmm6
        addsd xmm5, xmm6
        inc r9
        add r8, 8
        dec rcx
        jnz .remainder
    
    ; Find global maximum from 4 candidates in ymm1
    .find_global_max:
        ; Horizontal reduction for maximum
        vperm2f128 ymm6, ymm1, ymm1, 0x01  ; Swap halves
        vmaxpd ymm1, ymm1, ymm6            ; Max of 4 values -> 2 values
        
        ; Extract to XMM and find final max
        vextractf128 xmm6, ymm1, 1
        maxpd xmm1, xmm6                   ; Max of 2 values -> 1 value
        
        ; Get corresponding position
        movshdup xmm7, xmm1                 ; Duplicate
        maxpd xmm1, xmm7                    ; Final max value
        movsd xmm0, xmm1                    ; PCE = max value
    
    ; Compute mean and variance for PCE normalization
    ; Mean = sum / n
    vperm2f128 ymm6, ymm4, ymm4, 0x01
    vaddpd ymm4, ymm4, ymm6
    vextractf128 xmm6, ymm4, 1
    addpd xmm4, xmm6
    movshdup xmm6, xmm4
    addsd xmm4, xmm6                        ; Total sum
    
    mov rax, rsi
    imul rax, rdx                           ; n = width * height
    cvtsi2sd xmm6, rax
    divsd xmm4, xmm6                        ; Mean
    
    ; Variance = sum_sq / n - mean²
    vperm2f128 ymm6, ymm5, ymm5, 0x01
    vaddpd ymm5, ymm5, ymm6
    vextractf128 xmm6, ymm5, 1
    addpd xmm5, xmm6
    movshdup xmm6, xmm5
    addsd xmm5, xmm6                        ; Total sum of squares
    
    divsd xmm5, xmm6                        ; sum_sq / n
    movsd xmm7, xmm4
    mulsd xmm7, xmm7                        ; mean²
    subsd xmm5, xmm7                        ; Variance
    
    ; PCE = (peak - mean) / sqrt(variance)
    subsd xmm0, xmm4                        ; peak - mean
    vsqrtsd xmm5, xmm5, xmm5               ; sqrt(variance)
    divsd xmm0, xmm5                        ; PCE
    
    ; Compute p-value (proprietary statistical test)
    ; p-value = erfc(PCE / sqrt(2)) / 2
    ; Use fast approximation or call C math library
    movsd xmm1, [rel .sqrt2_inv]
    mulsd xmm0, xmm1
    ; Call erfc approximation (implemented separately)
    call wu_erfc_approx
    movsd xmm1, xmm0                        ; p-value
    
    ; Return PCE in xmm0 (already set)
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; Fast erfc approximation (proprietary)
wu_erfc_approx:
    ; Input: XMM0 = x
    ; Output: XMM0 = erfc(x)
    ; Uses polynomial approximation (proprietary coefficients)
    movsd xmm1, [rel .erfc_coeff_0]
    movsd xmm2, [rel .erfc_coeff_1]
    ; ... polynomial evaluation ...
    ret

section .data
align 32
.indices_init: dq 0, 1, 2, 3
.indices_step: dq 4, 4, 4, 4
.sqrt2_inv: dq 0.7071067811865476  ; 1/sqrt(2)
.erfc_coeff_0: dq 1.0
.erfc_coeff_1: dq -1.1283791670955126  ; -2/sqrt(pi)
```

**Optimization Details:**

1. **Double Precision Handling:**
   - AVX2 processes 4 doubles per iteration (vs 8 floats)
   - Use `vmovapd` and `vaddpd` for double precision

2. **Peak Detection:**
   - Track both value and position simultaneously
   - Use `vblendvpd` for conditional updates
   - Reduces branch mispredictions

3. **Statistical Computation:**
   - Compute mean and variance in parallel with peak search
   - Use FMA for sum of squares: `vfmadd231pd`
   - Fast erfc approximation for p-value

4. **Memory Access:**
   - Sequential access pattern (cache-friendly)
   - Prefetch 256 bytes ahead
   - Aligned loads when possible

---

### Function Architecture: Block Grid Offset Detection

#### C Interface
```c
void wu_detect_grid_offset_proprietary_avx2(
    const double* image,
    int width,
    int height,
    int* best_x_offset,    // Output: 0-7
    int* best_y_offset,    // Output: 0-7
    double* confidence     // Output: 0.0-1.0
);
```

#### Assembly Implementation Structure

**Algorithm:**
1. Test all 64 offset combinations (8x8 grid)
2. For each offset, compute blockiness score
3. Find maximum score
4. Compute confidence from score distribution

**Optimization Strategy:**
- Parallel offset testing (OpenMP in C wrapper)
- SIMD blockiness computation (assembly)
- Cache-friendly memory access

```asm
; blockgrid.asm - Blockiness computation for single offset
section .text
global wu_compute_blockiness_avx2

wu_compute_blockiness_avx2:
    ; Input: RDI = image (double precision), RSI = width, RDX = height
    ;        RCX = x_offset, R8 = y_offset
    ; Output: XMM0 = blockiness score
    
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    
    ; Save parameters
    mov r9, rdi             ; image pointer
    mov r10, rsi            ; width
    mov r11, rdx            ; height
    
    ; Initialize accumulators
    vxorpd ymm7, ymm7, ymm7  ; Sum of squared differences (vertical)
    vxorpd ymm8, ymm8, ymm8  ; Sum of squared differences (horizontal)
    mov r12, 0               ; Count of vertical boundaries
    mov r13, 0               ; Count of horizontal boundaries
    
    ; Compute vertical boundary differences
    ; Iterate over x = x_offset, x_offset+8, x_offset+16, ...
    mov rax, rcx             ; Start at x_offset
    .vertical_loop:
        cmp rax, r10
        jge .horizontal_start
        
        ; For each x, compare column x with column x+1
        ; Process 4 rows at a time (AVX2: 4 doubles)
        mov rbx, 0           ; y = 0
        .vertical_column:
            cmp rbx, r11
            jge .vertical_next_x
            
            ; Calculate addresses
            ; image[y * width + x] and image[y * width + x + 1]
            mov r14, rbx
            imul r14, r10    ; y * width
            add r14, rax     ; + x
            lea r15, [r14 + 1] ; x + 1
            
            ; Load 4 consecutive pixels from each column
            ; Column x: [y, y+1, y+2, y+3]
            mov rdx, r14
            imul rdx, 8      ; Convert to byte offset (double = 8 bytes)
            add rdx, r9      ; Add base pointer
            
            mov rsi, r15
            imul rsi, 8
            add rsi, r9
            
            ; Check if we have at least 4 rows remaining
            mov rcx, r11
            sub rcx, rbx
            cmp rcx, 4
            jl .vertical_scalar
            
            ; SIMD: Load 4 doubles from each column
            vmovupd ymm0, [rdx]      ; Column x (4 pixels)
            vmovupd ymm1, [rsi]      ; Column x+1 (4 pixels)
            
            ; Compute squared difference
            vsubpd ymm2, ymm0, ymm1  ; Difference
            vfmadd231pd ymm7, ymm2, ymm2  ; Accumulate squared diff (FMA)
            
            add rbx, 4
            add r12, 4
            jmp .vertical_column
        
        .vertical_scalar:
            ; Handle remaining 1-3 rows
            test rcx, rcx
            jz .vertical_next_x
            
            movsd xmm0, [rdx]
            movsd xmm1, [rsi]
            subsd xmm0, xmm1
            mulsd xmm0, xmm0
            addsd xmm7, xmm0
            
            add rdx, 8
            add rsi, 8
            inc rbx
            inc r12
            dec rcx
            jnz .vertical_scalar
        
        .vertical_next_x:
            add rax, 8       ; Next block boundary (8 pixels)
            jmp .vertical_loop
    
    ; Compute horizontal boundary differences
    .horizontal_start:
        mov rax, r8          ; Start at y_offset
        .horizontal_loop:
            cmp rax, r11
            jge .finalize
            
            ; For each y, compare row y with row y+1
            ; Process 4 columns at a time
            mov rbx, 0       ; x = 0
            .horizontal_row:
                cmp rbx, r10
                jge .horizontal_next_y
                
                ; Calculate addresses
                ; image[y * width + x] and image[(y+1) * width + x]
                mov r14, rax
                imul r14, r10    ; y * width
                add r14, rbx     ; + x
                
                mov r15, rax
                inc r15          ; y + 1
                imul r15, r10
                add r15, rbx
                
                ; Convert to byte offsets
                mov rdx, r14
                imul rdx, 8
                add rdx, r9
                
                mov rsi, r15
                imul rsi, 8
                add rsi, r9
                
                ; Check if we have at least 4 columns remaining
                mov rcx, r10
                sub rcx, rbx
                cmp rcx, 4
                jl .horizontal_scalar
                
                ; SIMD: Load 4 doubles from each row
                vmovupd ymm0, [rdx]      ; Row y (4 pixels)
                vmovupd ymm1, [rsi]      ; Row y+1 (4 pixels)
                
                ; Compute squared difference
                vsubpd ymm2, ymm0, ymm1
                vfmadd231pd ymm8, ymm2, ymm2
                
                add rbx, 4
                add r13, 4
                jmp .horizontal_row
            
            .horizontal_scalar:
                test rcx, rcx
                jz .horizontal_next_y
                
                movsd xmm0, [rdx]
                movsd xmm1, [rsi]
                subsd xmm0, xmm1
                mulsd xmm0, xmm0
                addsd xmm8, xmm0
                
                add rdx, 8
                add rsi, 8
                inc rbx
                inc r13
                dec rcx
                jnz .horizontal_scalar
            
            .horizontal_next_y:
                add rax, 8       ; Next block boundary
                jmp .horizontal_loop
    
    ; Finalize: Combine vertical and horizontal, compute average
    .finalize:
        ; Reduce vertical accumulator
        vperm2f128 ymm9, ymm7, ymm7, 0x01
        vaddpd ymm7, ymm7, ymm9
        vextractf128 xmm9, ymm7, 1
        addpd xmm7, xmm9
        movshdup xmm9, xmm7
        addsd xmm7, xmm9        ; Total vertical difference
        
        ; Reduce horizontal accumulator
        vperm2f128 ymm10, ymm8, ymm8, 0x01
        vaddpd ymm8, ymm8, ymm10
        vextractf128 xmm10, ymm8, 1
        addpd xmm8, xmm10
        movshdup xmm10, xmm8
        addsd xmm8, xmm10       ; Total horizontal difference
        
        ; Combine
        addsd xmm7, xmm8        ; Total difference
        
        ; Compute average
        mov rax, r12
        add rax, r13            ; Total count
        test rax, rax
        jz .zero_result
        
        cvtsi2sd xmm9, rax
        divsd xmm7, xmm9        ; Average = total / count
        movsd xmm0, xmm7
        jmp .done
    
    .zero_result:
        vxorpd xmm0, xmm0, xmm0
    
    .done:
        pop r14
        pop r13
        pop r12
        pop rbx
        pop rbp
        ret
```

**Optimization Details:**

1. **Cache-Friendly Access:**
   - Process columns vertically (sequential memory access)
   - Process rows horizontally (sequential memory access)
   - Prefetch next cache line when possible

2. **SIMD Processing:**
   - Process 4 doubles at a time (AVX2 width for doubles)
   - Use FMA for squared differences: `vfmadd231pd`
   - Reduces from 2 operations to 1

3. **Boundary Handling:**
   - Scalar fallback for remainder (1-3 elements)
   - Avoids complex masking operations

4. **Register Usage:**
   - Separate accumulators for vertical/horizontal
   - Allows parallel processing if needed

---

### Function Architecture: Lighting Direction Estimation

#### C Interface
```c
void wu_estimate_light_direction_avx2(
    const double* gray,
    const double* gx,
    const double* gy,
    int width,
    int height,
    double* azimuth,       // Output: 0-360 degrees
    double* elevation,     // Output: 0-90 degrees
    double* confidence     // Output: 0.0-1.0
);
```

#### Assembly Implementation Structure

**Key Operations:**
1. **Weighted Gradient Sum**: Weight by gradient magnitude
2. **Variance Computation**: For confidence calculation
3. **Angle Computation**: atan2 for azimuth

```asm
; lighting.asm - Light direction estimation
section .text
global wu_estimate_light_direction_avx2

wu_estimate_light_direction_avx2:
    ; Input: RDI = gray, RSI = gx, RDX = gy (all double precision)
    ;        RCX = width, R8 = height
    ; Output: XMM0 = azimuth, XMM1 = elevation, XMM2 = confidence
    
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    
    ; Save parameters
    mov r9, rdi             ; gray
    mov r10, rsi            ; gx
    mov r11, rdx            ; gy
    
    ; Initialize accumulators
    vxorpd ymm7, ymm7, ymm7  ; Weighted sum of gx
    vxorpd ymm8, ymm8, ymm8  ; Weighted sum of gy
    vxorpd ymm9, ymm9, ymm9  ; Total weight
    vxorpd ymm10, ymm10, ymm10 ; Sum of gradient magnitudes (for variance)
    
    ; Compute total pixels
    mov rax, rcx
    imul rax, r8            ; width * height
    mov r12, rax             ; Total pixels
    
    ; Find 25th percentile threshold (skip weak gradients)
    ; First pass: compute gradient magnitudes and find threshold
    mov r13, 0               ; Pixel index
    mov r14, r12
    shr r14, 2               ; n/4 for 25th percentile
    
    ; Quick pass to find threshold (simplified - use fixed threshold in practice)
    movsd xmm11, [rel .min_gradient_threshold]
    
    ; Main loop: Process 4 pixels at a time
    mov r13, 0
    .loop:
        cmp r13, r12
        jge .compute_angles
        
        ; Load gradients
        mov rax, r13
        imul rax, 8          ; Byte offset
        vmovupd ymm0, [r10 + rax]  ; gx (4 pixels)
        vmovupd ymm1, [r11 + rax]  ; gy (4 pixels)
        
        ; Compute gradient magnitude: sqrt(gx² + gy²)
        vmulpd ymm2, ymm0, ymm0     ; gx²
        vfmadd231pd ymm2, ymm1, ymm1 ; gx² + gy² (FMA)
        vsqrtpd ymm3, ymm2          ; magnitude = sqrt(gx² + gy²)
        
        ; Filter weak gradients (proprietary threshold)
        vbroadcastsd ymm4, xmm11     ; Threshold
        vcmppd ymm5, ymm3, ymm4, 1  ; magnitude > threshold
        
        ; Normalize weights (magnitude / max_magnitude)
        ; For simplicity, use magnitude directly as weight
        vandpd ymm6, ymm3, ymm5     ; Zero out weak gradients
        
        ; Accumulate weighted gradients
        vmulpd ymm0, ymm0, ymm6     ; gx * weight
        vmulpd ymm1, ymm1, ymm6     ; gy * weight
        vaddpd ymm7, ymm7, ymm0     ; Sum weighted gx
        vaddpd ymm8, ymm8, ymm1     ; Sum weighted gy
        vaddpd ymm9, ymm9, ymm6     ; Sum weights
        vaddpd ymm10, ymm10, ymm3   ; Sum magnitudes (for variance)
        
        add r13, 4
        jmp .loop
    
    ; Compute angles
    .compute_angles:
        ; Reduce accumulators
        vperm2f128 ymm11, ymm7, ymm7, 0x01
        vaddpd ymm7, ymm7, ymm11
        vextractf128 xmm11, ymm7, 1
        addpd xmm7, xmm11
        movshdup xmm11, xmm7
        addsd xmm7, xmm11           ; Weighted sum of gx
        
        vperm2f128 ymm11, ymm8, ymm8, 0x01
        vaddpd ymm8, ymm8, ymm11
        vextractf128 xmm11, ymm8, 1
        addpd xmm8, xmm11
        movshdup xmm11, xmm8
        addsd xmm8, xmm11           ; Weighted sum of gy
        
        vperm2f128 ymm11, ymm9, ymm9, 0x01
        vaddpd ymm9, ymm9, ymm11
        vextractf128 xmm11, ymm9, 1
        addpd xmm9, xmm11
        movshdup xmm11, xmm9
        addsd xmm9, xmm11           ; Total weight
        
        ; Normalize
        divsd xmm7, xmm9            ; Normalized gx
        divsd xmm8, xmm9            ; Normalized gy
        
        ; Compute azimuth: atan2(gy, gx)
        ; Note: atan2 requires C math library call
        ; For now, return normalized values, compute atan2 in C wrapper
        movsd xmm0, xmm7            ; Return normalized gx
        movsd xmm1, xmm8            ; Return normalized gy
        
        ; Compute confidence from variance
        ; Variance = E[magnitude²] - E[magnitude]²
        vperm2f128 ymm11, ymm10, ymm10, 0x01
        vaddpd ymm10, ymm10, ymm11
        vextractf128 xmm11, ymm10, 1
        addpd xmm10, xmm11
        movshdup xmm11, xmm10
        addsd xmm10, xmm11          ; Sum of magnitudes
        
        divsd xmm10, xmm9           ; Mean magnitude
        mulsd xmm10, xmm10          ; Mean²
        
        ; Confidence = 1 / (1 + variance)
        ; Simplified: use mean magnitude as proxy
        movsd xmm2, [rel .one]
        addsd xmm2, xmm10
        divsd xmm2, [rel .one]      ; 1 / (1 + mean²)
        
        pop r14
        pop r13
        pop r12
        pop rbx
        pop rbp
        ret

section .data
align 32
.min_gradient_threshold: dq 0.1    ; Minimum gradient magnitude
.one: dq 1.0
```

**Optimization Details:**

1. **Weighted Statistics:**
   - Weight by gradient magnitude (stronger gradients = more reliable)
   - Use FMA for efficient computation
   - Filter weak gradients to reduce noise

2. **Vectorized Operations:**
   - Process 4 doubles at a time
   - Compute magnitude, filter, and accumulate in parallel
   - Reduces loop overhead

3. **Statistical Computation:**
   - Compute weighted mean in single pass
   - Variance approximation for confidence
   - All operations vectorized

4. **Math Library Integration:**
   - `atan2` computed in C wrapper (calls libm)
   - Assembly handles all SIMD-accelerated operations
   - Clean separation of concerns

---

### Build System Architecture

#### Platform Detection and Compilation
```python
# build.py (enhanced)
import platform
import subprocess
import os

class AssemblyBuilder:
    def __init__(self):
        self.platform = platform.system()
        self.machine = platform.machine()
        self.arch = self._detect_architecture()
    
    def _detect_architecture(self):
        """Detect CPU architecture and SIMD capabilities."""
        if self.machine in ('x86_64', 'AMD64'):
            # Check for AVX2 support
            if self._check_avx2():
                return 'x86_64_avx2'
            elif self._check_sse2():
                return 'x86_64_sse2'
            return 'x86_64'
        elif self.machine in ('arm64', 'aarch64'):
            return 'arm64_neon'
        else:
            return 'generic'
    
    def build_assembly(self, asm_file, output_obj):
        """Assemble .asm file to .o based on platform."""
        if self.arch.startswith('x86_64'):
            return self._build_x86_64(asm_file, output_obj)
        elif self.arch == 'arm64_neon':
            return self._build_arm64(asm_file, output_obj)
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
    
    def _build_x86_64(self, asm_file, output_obj):
        """Build x86-64 assembly with YASM."""
        if self.platform == 'Windows':
            fmt = 'win64'
        elif self.platform == 'Darwin':
            fmt = 'macho64'
        else:
            fmt = 'elf64'
        
        cmd = [
            'yasm',
            '-f', fmt,
            '-o', output_obj,
            asm_file
        ]
        return subprocess.run(cmd, check=True)
    
    def _build_arm64(self, asm_file, output_obj):
        """Build ARM64 assembly with GAS."""
        cmd = [
            'as',
            '-64',
            '-o', output_obj,
            asm_file
        ]
        return subprocess.run(cmd, check=True)
```

#### Multi-Platform Build Matrix
```
Platform          | Architecture | Assembler | SIMD      | Output
------------------|--------------|-----------|-----------|----------
Windows x64       | x86_64       | YASM      | AVX2      | .dll
Linux x64         | x86_64       | YASM      | AVX2      | .so
macOS Intel       | x86_64       | YASM      | AVX2      | .dylib
macOS Apple Silicon| arm64       | GAS       | NEON      | .dylib
Linux ARM64       | arm64        | GAS       | NEON      | .so
```

---

### Common Utility Functions

**Horizontal Reduction (Optimized):**

```asm
; common.asm - Shared utility functions

; Horizontal sum of 8 floats (AVX2)
; Input: YMM0 = [a0, a1, a2, a3, a4, a5, a6, a7]
; Output: XMM0 = sum
; Based on Intel Optimization Manual recommendations
section .text
global horizontal_sum_avx2_fast

horizontal_sum_avx2_fast:
    ; Method 1: vperm2f128 + vaddps (faster than vhaddps)
    vperm2f128 ymm1, ymm0, ymm0, 0x01  ; Swap high/low halves
    vaddps ymm0, ymm0, ymm1            ; Add: [a0+a4, a1+a5, a2+a6, a3+a7, ...]
    
    ; Extract to XMM and reduce further
    vextractf128 xmm1, ymm0, 1         ; Extract high 128 bits
    addps xmm0, xmm1                   ; Add: [a0+a4+a2+a6, a1+a5+a3+a7, ...]
    
    ; Final reduction using movshdup (faster than haddps)
    movshdup xmm1, xmm0                ; Duplicate high elements
    addss xmm0, xmm1                   ; Final sum
    ret

; Horizontal sum of 4 doubles (AVX2)
; Input: YMM0 = [a0, a1, a2, a3]
; Output: XMM0 = sum
global horizontal_sum_double_avx2

horizontal_sum_double_avx2:
    vperm2f128 ymm1, ymm0, ymm0, 0x01
    vaddpd ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    addpd xmm0, xmm1
    movshdup xmm1, xmm0
    addsd xmm0, xmm1
    ret

; Fast square root approximation (for magnitude computation)
; Uses Newton-Raphson iteration (2 iterations sufficient for float precision)
; Input: XMM0 = x (positive)
; Output: XMM0 = sqrt(x)
global fast_sqrt_approx

fast_sqrt_approx:
    ; Initial guess: rsqrt(x) using hardware instruction
    rsqrtss xmm1, xmm0                 ; 1/sqrt(x)
    mulss xmm1, xmm0                   ; sqrt(x) approximation
    
    ; One Newton-Raphson iteration: x_new = 0.5 * (x_old + x / x_old)
    divss xmm2, xmm0, xmm1            ; x / x_old
    addss xmm2, xmm1                  ; x_old + x/x_old
    mulss xmm2, [rel .half]           ; 0.5 * (...)
    movss xmm0, xmm2
    ret

section .data
.half: dd 0.5
```

**Performance Tuning Tips (From Intel Optimization Manual):**

1. **Instruction Scheduling:**
   - Separate dependent instructions by 3-4 instructions
   - Use multiple accumulators to break dependency chains
   - Example: Use ymm7 and ymm8 instead of just ymm7

2. **Memory Access Patterns:**
   - Aligned loads (`vmovaps`) are 2x faster than unaligned (`vmovups`)
   - Prefetch 256-512 bytes ahead for sequential access
   - Use `prefetchnta` for streaming data (bypasses cache)

3. **FMA vs Separate Multiply-Add:**
   - FMA (`vfmadd231ps`): 1 cycle latency, 2 per cycle throughput
   - Separate (`vmulps` + `vaddps`): 2 cycles latency, 1 per cycle throughput
   - **Always prefer FMA when available**

4. **Horizontal Reductions:**
   - `vhaddps`: 3 cycles latency, 1 per cycle throughput
   - Manual (`vperm2f128` + `vaddps`): 2 cycles latency, 1 per cycle throughput
   - **Manual reduction is faster for small vectors**

5. **Loop Unrolling:**
   - Unroll 4-8x for best performance
   - Reduces branch misprediction penalty
   - Allows better instruction scheduling
   - Balance: Too much unrolling increases code size (I-cache pressure)

6. **Register Allocation:**
   - x86-64 has 16 YMM registers (ymm0-ymm15)
   - Use 8-10 for computation, reserve rest for constants/temporaries
   - Avoid register spilling (accessing stack)

7. **Cache Optimization:**
   - L1 cache: 32KB, 8-way associative
   - L2 cache: 256KB per core
   - Process data in 64KB chunks (fits in L1)
   - Use `prefetchnta` for data used once (streaming)

---

### Memory Architecture

#### Data Alignment Strategy
```c
// All SIMD data must be 32-byte aligned (AVX2 requirement)
#define ALIGN_32 __attribute__((aligned(32)))

// Pre-allocated buffers
typedef struct {
    float* features;      // ALIGN_32
    int* positions;       // ALIGN_32
    float* matches;       // ALIGN_32
} CopyMoveBuffers;

// Allocation
CopyMoveBuffers* buffers = _mm_malloc(sizeof(CopyMoveBuffers), 32);
buffers->features = _mm_malloc(n_blocks * n_features * sizeof(float), 32);
```

#### Cache Optimization
- **Block size**: 64KB chunks (fits in L1 cache)
- **Prefetching**: Use `_mm_prefetch` for next iteration
- **Memory layout**: Structure of Arrays (SoA) for SIMD

---

### Error Handling Architecture

#### Assembly Error Codes
```c
typedef enum {
    WU_SUCCESS = 0,
    WU_ERROR_NULL_POINTER = -1,
    WU_ERROR_INVALID_SIZE = -2,
    WU_ERROR_ALIGNMENT = -3,
    WU_ERROR_SIMD_NOT_AVAILABLE = -4
} wu_error_t;
```

#### Validation in C Wrapper
```c
double wu_similarity_match_proprietary(
    const float* features_a,
    const float* features_b,
    size_t n_features
) {
    // Validate inputs
    if (!features_a || !features_b) {
        return NAN;  // Error indicator
    }
    if (n_features == 0) {
        return 0.0;
    }
    
    // Check alignment
    if ((uintptr_t)features_a % 32 != 0 || 
        (uintptr_t)features_b % 32 != 0) {
        // Fallback to unaligned version or error
        return wu_similarity_match_unaligned(features_a, features_b, n_features);
    }
    
    // Call assembly function
    return wu_similarity_match_proprietary_avx2(features_a, features_b, n_features);
}
```

---

### Runtime Execution Architecture

#### Function Call Flow
```
Python Code
    │
    │ import wu.native.simd
    ▼
Python ctypes Binding (simd.py)
    │
    │ ctypes.CDLL('wu_core.dll')
    ▼
C Wrapper Function (copymove_wrapper.c)
    │
    │ - Validate inputs
    │ - Check alignment
    │ - Handle errors
    ▼
Assembly Function (copymove.asm)
    │
    │ - SIMD operations
    │ - Proprietary algorithm
    │ - Obfuscated code
    ▼
Return Result (XMM0 register)
    │
    │ ctypes.c_double
    ▼
Python float
```

#### SIMD Capability Detection
```c
// Runtime SIMD detection (in C wrapper)
typedef enum {
    SIMD_NONE = 0,
    SIMD_SSE2 = 1,
    SIMD_AVX = 2,
    SIMD_AVX2 = 4,
    SIMD_AVX512 = 8,
    SIMD_NEON = 16
} simd_cap_t;

simd_cap_t detect_simd_caps(void) {
    simd_cap_t caps = SIMD_NONE;
    
#ifdef __x86_64__
    // CPUID check for AVX2
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    if (ecx & (1 << 28)) caps |= SIMD_AVX;
    
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (ebx & (1 << 5)) caps |= SIMD_AVX2;
#endif

#ifdef __aarch64__
    caps |= SIMD_NEON;  // Always available on ARM64
#endif

    return caps;
}

// Function dispatch based on capabilities
double wu_similarity_match_proprietary(...) {
    simd_cap_t caps = detect_simd_caps();
    
    if (caps & SIMD_AVX2) {
        return wu_similarity_match_proprietary_avx2(...);
    } else if (caps & SIMD_SSE2) {
        return wu_similarity_match_proprietary_sse2(...);
    } else {
        return wu_similarity_match_proprietary_scalar(...);
    }
}
```

#### Memory Management Architecture
```
┌─────────────────────────────────────────┐
│         Python NumPy Array               │
│  (numpy.ndarray, may be unaligned)      │
└──────────────┬──────────────────────────┘
               │
               │ np.ascontiguousarray()
               ▼
┌─────────────────────────────────────────┐
│      Contiguous NumPy Array              │
│  (aligned to 8-byte boundary)            │
└──────────────┬──────────────────────────┘
               │
               │ Check alignment
               │ If not 32-byte aligned:
               ▼
┌─────────────────────────────────────────┐
│      Temporary Aligned Buffer           │
│  (_mm_malloc, 32-byte aligned)           │
│  - Copy data                             │
│  - Process                               │
│  - Copy back                             │
└──────────────┬──────────────────────────┘
               │
               │ If 32-byte aligned:
               ▼
┌─────────────────────────────────────────┐
│      Direct SIMD Processing              │
│  (no copy needed)                        │
└──────────────────────────────────────────┘
```

---

### Testing Architecture

#### Unit Test Structure
```python
# tests/test_assembly.py
import numpy as np
from wu.native import simd

def test_similarity_match():
    # Generate test data
    a = np.random.rand(128).astype(np.float32)
    b = np.random.rand(128).astype(np.float32)
    
    # Ensure alignment
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    
    # Test assembly implementation
    result_asm = simd.similarity_match_proprietary(a, b)
    
    # Test reference (Python/NumPy)
    result_ref = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Compare (allow small floating-point differences)
    assert abs(result_asm - result_ref) < 1e-5
```

#### Performance Benchmarks
```python
# benchmarks/benchmark_assembly.py
import timeit
import numpy as np
from wu.native import simd

def benchmark_similarity():
    # Setup: Create aligned test data
    n = 1024
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    
    # Ensure alignment
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    
    # Warmup
    _ = simd.similarity_match_proprietary(a, b)
    
    # Assembly version
    time_asm = timeit.timeit(
        lambda: simd.similarity_match_proprietary(a, b),
        number=10000
    )
    
    # NumPy reference
    def numpy_version():
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    time_numpy = timeit.timeit(numpy_version, number=10000)
    
    speedup = time_numpy / time_asm
    print(f"Speedup: {speedup:.2f}x")
    print(f"Assembly: {time_asm*1000:.2f}ms")
    print(f"NumPy: {time_numpy*1000:.2f}ms")
```

#### Debugging Assembly Code

**1. Use Debugger (GDB/LLDB):**
```bash
# Compile with debug symbols (keep in private build)
yasm -g dwarf2 -f elf64 -o copymove.o copymove.asm
gcc -g -shared -o libwu_core.so copymove.o

# Debug
gdb python
(gdb) break wu_similarity_match_proprietary_avx2
(gdb) run test_assembly.py
(gdb) info registers ymm0 ymm1 ymm7 ymm8
(gdb) x/8f $rdi  # Examine memory at RDI
```

**2. Validation Against Reference:**
```python
# tests/test_assembly_correctness.py
def test_similarity_correctness():
    """Test assembly against NumPy reference."""
    for n in [16, 64, 128, 256, 512, 1024]:
        a = np.random.rand(n).astype(np.float32)
        b = np.random.rand(n).astype(np.float32)
        
        result_asm = simd.similarity_match_proprietary(a, b)
        result_ref = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Allow small floating-point differences
        assert abs(result_asm - result_ref) < 1e-5, \
            f"Mismatch at n={n}: {result_asm} vs {result_ref}"
```

**3. Performance Profiling:**
```bash
# Linux: Use perf
perf record -g python test_assembly.py
perf report

# Check SIMD utilization
perf stat -e instructions,cycles,avx_insts.executed python test_assembly.py

# Windows: Use Intel VTune
vtune -collect hotspots -result-dir ./vtune_results python test_assembly.py
```

**4. Memory Alignment Validation:**
```c
// c_wrappers/copymove_wrapper.c
double wu_similarity_match_proprietary(
    const float* features_a,
    const float* features_b,
    size_t n_features
) {
    // Check alignment
    uintptr_t addr_a = (uintptr_t)features_a;
    uintptr_t addr_b = (uintptr_t)features_b;
    
    if (addr_a % 32 != 0 || addr_b % 32 != 0) {
        // Fallback to unaligned version or error
        fprintf(stderr, "Warning: Unaligned memory (a=%p, b=%p)\n", 
                features_a, features_b);
        return wu_similarity_match_unaligned(features_a, features_b, n_features);
    }
    
    return wu_similarity_match_proprietary_avx2(features_a, features_b, n_features);
}
```

**5. Assembly Code Instrumentation:**
```asm
; Add timing markers for profiling
wu_similarity_match_proprietary_avx2:
    ; Start timing (if profiling enabled)
    %ifdef PROFILE
        rdtsc
        mov [rel .start_time], eax
    %endif
    
    ; ... main computation ...
    
    ; End timing
    %ifdef PROFILE
        rdtsc
        sub eax, [rel .start_time]
        mov [rel .cycle_count], eax
    %endif
    
    ret

%ifdef PROFILE
section .data
.start_time: dd 0
.cycle_count: dd 0
%endif
```

---

## Technical Recommendations

### 1. Memory Management

**Best Practices:**
- Use aligned allocations (`aligned_alloc`, `_mm_malloc`) for SIMD
- Pre-allocate buffers to avoid repeated malloc/free
- Use memory pools for temporary allocations
- Consider zero-copy when possible (NumPy array views)

**Example:**
```c
// Pre-allocate feature buffer for copy-move
float* features = _mm_malloc(n_blocks * n_features * sizeof(float), 32);
// ... use features ...
_mm_free(features);
```

### 2. Compiler Optimizations

**Build flags:**
```bash
# GCC/Clang
-O3 -march=native -mtune=native -mavx2 -mfma -fopenmp

# MSVC
/O2 /arch:AVX2 /openmp
```

**Why `-march=native`:**
- Enables all CPU-specific optimizations
- Uses FMA (fused multiply-add) when available
- May enable AVX-512 on newer CPUs

### 3. Parallelization Strategy

**OpenMP Guidelines:**
- Use `#pragma omp parallel for` for independent iterations
- Use `schedule(dynamic)` for uneven workloads
- Use `reduction` for accumulations
- Avoid false sharing (pad shared data structures)

**Example:**
```c
#pragma omp parallel for schedule(dynamic) reduction(+:total_diff)
for (int i = 0; i < n_blocks; i++) {
    // Process block i independently
    total_diff += process_block(blocks[i]);
}
```

### 4. SIMD Usage Patterns

**Do:**
- Process data in multiples of SIMD width (8 floats for AVX2)
- Align data to 32-byte boundaries
- Use `_mm256_loadu_ps` for unaligned, `_mm256_load_ps` for aligned
- Unroll small loops manually

**Don't:**
- Mix SIMD and scalar in hot loops
- Use gather/scatter instructions (slow on most CPUs)
- Assume data is aligned without checking

### 5. Profiling and Measurement

**Tools:**
- **Linux**: `perf record` / `perf report`
- **Windows**: Intel VTune, Windows Performance Analyzer
- **macOS**: Instruments (Time Profiler)
- **Python**: `cProfile`, `line_profiler`

**Key metrics:**
- Cycles per instruction (CPI)
- Cache miss rates (L1, L2, L3)
- SIMD utilization
- Memory bandwidth

**Example profiling:**
```bash
# Profile copy-move detection
perf record -g python -m cProfile -o profile.stats test_copymove.py
perf report
```

---

## Expected Performance Gains

### Single Image (Baseline: Current Python)
- Copy-move: **15-30x faster**
- Block grid: **20-40x faster**
- Lighting: **10-15x faster**
- PRNU: **20-50x faster**

### Video Processing (5 min @ 30fps = 9,000 frames)
**Current:** ~2-4 hours (estimated)
**After Phase 1-2:** ~10-20 minutes
**After Phase 3-4:** ~3-5 minutes

**Breakdown:**
- Frame extraction: 5-10x faster
- Per-frame analysis: 15-30x faster (aggregate)
- Batch processing: 2-3x additional speedup from parallelism

---

## Risk Mitigation

### 1. Correctness Validation
- **Unit tests**: Compare C output to Python reference
- **Floating-point tolerance**: Account for SIMD rounding differences
- **Edge cases**: Empty images, single-pixel images, etc.

### 2. Portability
- **Runtime detection**: Check SIMD capabilities before use
- **Fallbacks**: Scalar implementations for older CPUs
- **Cross-platform**: Test on Windows, Linux, macOS

### 3. Maintenance
- **Documentation**: Comment assembly code extensively
- **Version control**: Tag SIMD versions for different CPUs
- **Benchmarking**: Continuous performance regression testing

---

## Dependencies

### Required Libraries
- **FFTW3**: Fast Fourier Transform (PRNU, DCT)
- **OpenMP**: Parallel processing
- **FFmpeg**: Video frame extraction (optional, for video pipeline)
- **NASM/YASM**: Assembler for x86-64 assembly code
- **GAS**: GNU Assembler (for ARM64 assembly)

### Optional (for maximum performance)
- **Intel MKL**: Optimized BLAS/DCT (commercial license)
- **Intel IPP**: Image processing primitives (commercial license)

### Build Tools for Assembly
- **NASM** (Netwide Assembler): For x86-64 assembly
- **YASM**: Alternative x86-64 assembler
- **GAS** (GNU Assembler): For ARM64 assembly
- **objcopy**: For symbol stripping and obfuscation

---

## Next Steps

1. **Immediate (This Week):**
   - Set up build system for C extensions and assembly
   - Create `copymove_native.c` stub with function signatures
   - Set up NASM/YASM in build pipeline
   - Benchmark current Python implementation as baseline
   - **Decision**: Identify which algorithms are "secret sauce" worth protecting

2. **Short-term (Next 2 Weeks):**
   - Implement batch DCT extraction in C (reference implementation)
   - Add OpenMP parallelization
   - **Start assembly implementation** for top 2-3 critical algorithms
   - Validate correctness against Python

3. **Medium-term (Next Month):**
   - Complete Phase 1-2 implementations
   - **Complete assembly implementations** for core forensic algorithms
   - Add obfuscation techniques
   - Set up binary distribution (separate source from binaries)
   - Add video frame extraction
   - Benchmark on real video files

4. **Long-term (Next Quarter):**
   - Profile and optimize assembly code
   - Create platform-specific builds (x86-64, ARM64)
   - Set up CI/CD for binary distribution
   - Document public API while keeping algorithm details private
   - Consider license updates to clarify IP protection strategy

---

## Questions to Consider

1. **Target hardware**: What CPUs will this run on? (affects SIMD choice)
2. **Accuracy requirements**: Can we trade some accuracy for speed?
3. **Memory constraints**: How much RAM is available?
4. **Deployment**: Will this be distributed as binaries or source?
5. **IP Protection Level**: Which algorithms are truly proprietary vs. standard techniques?
6. **License Strategy**: How to balance open source with IP protection?
7. **Security Auditing**: Will you provide assembly source for security reviews?

---

## References

### Online Resources
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Agner Fog's Optimization Manuals: https://www.agner.org/optimize/
- FFTW Documentation: http://www.fftw.org/fftw3_doc/
- OpenMP Specification: https://www.openmp.org/specifications/
- NASM Documentation: https://www.nasm.us/docs.php
- x86-64 Assembly Reference: https://www.felixcloutier.com/x86/

### Local Assembly Documentation (Hopper Project)
**Location:** `c:/dev/hopper/Documents/Assembler/`

#### x86-64 (Intel)
- `x86_Intel/Intel_AVX_Introduction.pdf` - AVX/AVX2 intrinsics and programming
- `x86_Intel/Intel_Optimization_Manual_Vol1.pdf` - Performance optimization techniques
- `x86_Intel/Intel_Optimization_Manual_Vol2.pdf` - Advanced optimization strategies
- `x86_Intel/Intel_SDM_Combined_Vol1-4.pdf` - Complete x86-64 instruction set reference

#### x86-64 (AMD)
- `x86_AMD/AMD64_Vol1_Application_Programming.pdf` - AMD64 programming guide
- `x86_AMD/AMD64_Vol3_Instructions.pdf` - AMD64 instruction reference

#### Assembler Tools
- `FFmpeg_SIMD/YASM_User_Manual.pdf` - YASM assembler syntax and usage

#### ARM64/NEON
- `ARM/ARMv8-A_Architecture_Reference_Manual.pdf` - ARM64 architecture and NEON SIMD
- `ARM/ARMv9_Supplement.pdf` - ARMv9 enhancements

**Usage:** These manuals provide authoritative reference for:
- Instruction encodings and semantics
- Performance characteristics and optimization guidelines
- SIMD register usage and data alignment
- Platform-specific optimization techniques

## IP Protection Best Practices

### What Assembly Protects
✅ **Implementation details**: Your specific algorithm optimizations
✅ **Heuristics and thresholds**: Proprietary parameter tuning
✅ **Code structure**: Control flow and data organization
✅ **Performance tricks**: Hand-optimized instruction sequences
✅ **Obfuscation**: Makes reverse engineering significantly harder (10-100x effort)

### What Assembly Doesn't Protect
❌ **Algorithm concepts**: The general approach (can be reimplemented)
❌ **Patents**: Assembly doesn't help with patent claims
❌ **Mathematical formulas**: If the math is published, it's not protected
❌ **Determined reverse engineers**: Skilled analysts can still understand assembly (but it's much harder)

### Recommended Approach
1. **Protect the "Secret Sauce"**: Your unique heuristics, thresholds, and optimizations
2. **Document the API**: Keep Python interface open and well-documented
3. **Provide Binary Distribution**: Distribute compiled libraries, not assembly source
4. **Security Auditing**: Consider providing assembly source under NDA for security reviews
5. **License Clarity**: Update license to explain IP protection strategy

### Practical Obfuscation Techniques

**1. Register Obfuscation**
```asm
; Instead of obvious: mov rax, rdi
; Use: mov r10, rdi; mov rax, r10 (adds indirection)
; Or: lea rax, [rdi + 0] (equivalent but less obvious)
```

**2. Dummy Operations**
```asm
; Insert operations that don't affect results
vxorps ymm15, ymm15, ymm15  ; Clear register (looks important, but unused)
vaddps ymm0, ymm0, ymm0     ; Double (then halve later, net zero)
vmulps ymm1, ymm1, ymm1     ; Square (then sqrt later, net identity)
```

**3. Equivalent Instruction Sequences**
```asm
; Instead of: add rax, 1
; Use: lea rax, [rax + 1]
; Or: inc rax (different encoding, same result)

; Instead of: mov rax, 0
; Use: xor rax, rax (smaller, faster, less obvious)
```

**4. Control Flow Obfuscation**
```asm
; Instead of direct jumps, use computed jumps
lea r10, [rel .target]
jmp r10
; Makes static analysis harder
```

**5. Data Interleaving**
```asm
; Mix actual data with dummy constants
; Makes it harder to identify what's important
.data
    real_threshold: dd 0.9996
    dummy1: dd 0.1234
    real_coefficient: dd 0.587
    dummy2: dd 0.5678
```

**6. Symbol Stripping**
```bash
# Build with minimal symbols
gcc -O3 -fno-ident -s -o libwu_core.so wu_core.o

# Strip remaining symbols
strip -s libwu_core.so

# Or use objcopy for more control
objcopy --strip-all --strip-debug libwu_core.so
```

**7. Function Name Obfuscation**
```c
// Instead of: wu_similarity_match
// Use: wu_f7a3b2c1d4e5  (random hash-like names)
// Or: _Z12internal_xyz (mangled C++ names)
```

**8. Platform-Specific Code Splitting**
- Keep x86-64 and ARM64 implementations in separate files
- Makes it harder to compare algorithms across platforms
- Use different obfuscation techniques per platform

### Example License Addition
```
Core forensic algorithms are implemented in optimized assembly for performance
and IP protection. While the Python API is open source, the low-level
implementations are distributed as compiled binaries. Assembly source code
may be available for security auditing under appropriate agreements.
```

### Distribution Strategy

**Public Repository (GitHub):**
- ✅ Python API code (fully open)
- ✅ C utility functions (less critical)
- ✅ Documentation and examples
- ✅ Test suites
- ❌ Assembly source code (keep private)
- ❌ Compiled binaries (distribute via releases)

**Private Repository:**
- ✅ Assembly source code
- ✅ Build scripts and tooling
- ✅ Internal documentation
- ✅ Performance benchmarks

**Binary Distribution:**
- Release compiled `.so`/`.dll`/`.dylib` files via GitHub Releases
- Tag releases with platform (e.g., `wu-forensics-1.0.0-x86_64-linux.so`)
- Provide checksums for verification
- Consider code signing for additional trust

### Legal Considerations

**Copyright Protection:**
- Assembly code is protected by copyright (same as any code)
- Binary distribution is standard practice (many open source projects do this)
- You're not hiding the API, just the implementation

**Patent Considerations:**
- Assembly doesn't protect against patent claims
- If your algorithm is novel, consider patent protection separately
- Assembly makes it harder for competitors to copy, but doesn't prevent independent invention

**Open Source Compliance:**
- Your Python API remains fully open source
- Users can see what the code does (API level)
- Users can't easily copy how you do it (implementation level)
- This is similar to how many commercial products work (open API, closed implementation)

