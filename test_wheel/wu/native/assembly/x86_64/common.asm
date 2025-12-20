; Wu Forensics - Common Assembly Utilities
; x86-64 AVX2/FMA implementation
;
; This file contains shared utility functions used by other assembly modules.
; Target: x86-64 with AVX2 and FMA support
;
; Build (Windows): nasm -f win64 -o common.obj common.asm
; Build (Linux):   nasm -f elf64 -o common.o common.asm
; Build (macOS):   nasm -f macho64 -o common.o common.asm

; Platform-specific section directive
%ifidn __OUTPUT_FORMAT__, win64
    %define SECTION_TEXT section .text
    %define SECTION_DATA section .data align=32
    %define SECTION_RODATA section .rdata align=32
%elifidn __OUTPUT_FORMAT__, macho64
    %define SECTION_TEXT section .text
    %define SECTION_DATA section .data align=32
    %define SECTION_RODATA section .rodata align=32
%else
    %define SECTION_TEXT section .text
    %define SECTION_DATA section .data align=32
    %define SECTION_RODATA section .rodata align=32
%endif

; Windows x64 calling convention: RCX, RDX, R8, R9, then stack
; System V AMD64 (Linux/macOS): RDI, RSI, RDX, RCX, R8, R9, then stack

%ifidn __OUTPUT_FORMAT__, win64
    %define ARG1 rcx
    %define ARG2 rdx
    %define ARG3 r8
    %define ARG4 r9
%else
    %define ARG1 rdi
    %define ARG2 rsi
    %define ARG3 rdx
    %define ARG4 rcx
%endif

SECTION_TEXT

; ============================================================================
; HORIZONTAL SUM - Float (8 floats in YMM -> single float)
; ============================================================================
; Input:  YMM0 = [a0, a1, a2, a3, a4, a5, a6, a7]
; Output: XMM0 = sum of all 8 floats
; Clobbers: YMM1
;
; This is optimized based on Intel optimization manual recommendations:
; Uses vperm2f128 + vaddps (faster than vhaddps)
; ============================================================================
global wu_asm_hsum_f32
wu_asm_hsum_f32:
    ; Swap high/low 128-bit halves
    vperm2f128 ymm1, ymm0, ymm0, 0x01
    ; Add halves: [a0+a4, a1+a5, a2+a6, a3+a7, ...]
    vaddps ymm0, ymm0, ymm1
    
    ; Extract high 128 bits and add to low
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    
    ; Final reduction using movshdup (duplicate odd elements)
    movshdup xmm1, xmm0
    addss xmm0, xmm1
    
    ; Zero upper bits of YMM (AVX-SSE transition penalty avoidance)
    vzeroupper
    ret

; ============================================================================
; HORIZONTAL SUM - Double (4 doubles in YMM -> single double)
; ============================================================================
; Input:  YMM0 = [a0, a1, a2, a3]
; Output: XMM0 = sum of all 4 doubles
; Clobbers: YMM1
; ============================================================================
global wu_asm_hsum_f64
wu_asm_hsum_f64:
    ; Swap high/low 128-bit halves
    vperm2f128 ymm1, ymm0, ymm0, 0x01
    ; Add halves: [a0+a2, a1+a3, ...]
    vaddpd ymm0, ymm0, ymm1
    
    ; Extract high 128 bits
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    
    ; Final horizontal add for 2 doubles
    vhaddpd xmm0, xmm0, xmm0
    
    vzeroupper
    ret

; ============================================================================
; HORIZONTAL MAX - Float (8 floats in YMM -> single max float)
; ============================================================================
; Input:  YMM0 = [a0, a1, a2, a3, a4, a5, a6, a7]
; Output: XMM0 = maximum of all 8 floats
; Clobbers: YMM1
; ============================================================================
global wu_asm_hmax_f32
wu_asm_hmax_f32:
    vperm2f128 ymm1, ymm0, ymm0, 0x01
    vmaxps ymm0, ymm0, ymm1
    
    vextractf128 xmm1, ymm0, 1
    vmaxps xmm0, xmm0, xmm1
    
    movshdup xmm1, xmm0
    vmaxss xmm0, xmm0, xmm1
    
    vzeroupper
    ret

; ============================================================================
; DOT PRODUCT - Float32 (proprietary optimized version)
; ============================================================================
; This implementation includes loop unrolling and prefetching for maximum
; throughput on large arrays.
;
; Input:  ARG1 = pointer to float array A
;         ARG2 = pointer to float array B
;         ARG3 = count (number of floats)
; Output: XMM0 = dot product (as double for precision)
; ============================================================================
global wu_asm_dot_f32
wu_asm_dot_f32:
    push rbp
    mov rbp, rsp
    push rbx
    
    ; Move arguments to stable registers
    mov r10, ARG1           ; A pointer
    mov r11, ARG2           ; B pointer
    mov rcx, ARG3           ; count
    
    ; Initialize accumulators (multiple for better ILP)
    vxorps ymm0, ymm0, ymm0 ; acc0
    vxorps ymm1, ymm1, ymm1 ; acc1
    vxorps ymm2, ymm2, ymm2 ; acc2
    vxorps ymm3, ymm3, ymm3 ; acc3
    
    ; Check if we have at least 32 elements for unrolled loop
    cmp rcx, 32
    jl .small_loop
    
    ; Prefetch first cache lines
    prefetchnta [r10 + 256]
    prefetchnta [r11 + 256]
    
.unrolled_loop:
    ; Process 32 floats per iteration (4x8 = 32)
    ; Iteration 1
    vmovups ymm4, [r10]
    vmovups ymm5, [r11]
    vfmadd231ps ymm0, ymm4, ymm5
    
    ; Iteration 2
    vmovups ymm6, [r10 + 32]
    vmovups ymm7, [r11 + 32]
    vfmadd231ps ymm1, ymm6, ymm7
    
    ; Iteration 3
    vmovups ymm4, [r10 + 64]
    vmovups ymm5, [r11 + 64]
    vfmadd231ps ymm2, ymm4, ymm5
    
    ; Iteration 4
    vmovups ymm6, [r10 + 96]
    vmovups ymm7, [r11 + 96]
    vfmadd231ps ymm3, ymm6, ymm7
    
    ; Prefetch next iteration
    prefetchnta [r10 + 384]
    prefetchnta [r11 + 384]
    
    ; Advance pointers
    add r10, 128
    add r11, 128
    sub rcx, 32
    
    cmp rcx, 32
    jge .unrolled_loop

.small_loop:
    ; Combine accumulators
    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm0, ymm0, ymm2
    
    ; Process remaining 8 at a time
    cmp rcx, 8
    jl .scalar_loop
    
.vec8_loop:
    vmovups ymm4, [r10]
    vmovups ymm5, [r11]
    vfmadd231ps ymm0, ymm4, ymm5
    
    add r10, 32
    add r11, 32
    sub rcx, 8
    
    cmp rcx, 8
    jge .vec8_loop

.scalar_loop:
    ; Horizontal sum of ymm0
    call wu_asm_hsum_f32
    
    ; Process remaining 1-7 elements
    test rcx, rcx
    jz .done
    
    ; Scalar remainder
    vxorps xmm1, xmm1, xmm1
.scalar_remainder:
    vmovss xmm2, [r10]
    vmovss xmm3, [r11]
    vfmadd231ss xmm1, xmm2, xmm3
    
    add r10, 4
    add r11, 4
    dec rcx
    jnz .scalar_remainder
    
    ; Add scalar remainder to result
    vaddss xmm0, xmm0, xmm1

.done:
    ; Convert to double for precision
    vcvtss2sd xmm0, xmm0, xmm0
    
    vzeroupper
    pop rbx
    pop rbp
    ret

; ============================================================================
; DOT PRODUCT - Float64 (double precision)
; ============================================================================
global wu_asm_dot_f64
wu_asm_dot_f64:
    push rbp
    mov rbp, rsp
    
    mov r10, ARG1
    mov r11, ARG2
    mov rcx, ARG3
    
    vxorpd ymm0, ymm0, ymm0
    vxorpd ymm1, ymm1, ymm1
    
    cmp rcx, 8
    jl .f64_small
    
.f64_unrolled:
    vmovupd ymm2, [r10]
    vmovupd ymm3, [r11]
    vfmadd231pd ymm0, ymm2, ymm3
    
    vmovupd ymm4, [r10 + 32]
    vmovupd ymm5, [r11 + 32]
    vfmadd231pd ymm1, ymm4, ymm5
    
    add r10, 64
    add r11, 64
    sub rcx, 8
    
    cmp rcx, 8
    jge .f64_unrolled
    
    vaddpd ymm0, ymm0, ymm1

.f64_small:
    cmp rcx, 4
    jl .f64_scalar
    
    vmovupd ymm2, [r10]
    vmovupd ymm3, [r11]
    vfmadd231pd ymm0, ymm2, ymm3
    
    add r10, 32
    add r11, 32
    sub rcx, 4

.f64_scalar:
    ; Horizontal sum
    call wu_asm_hsum_f64
    
    test rcx, rcx
    jz .f64_done
    
.f64_remainder:
    vmovsd xmm2, [r10]
    vmovsd xmm3, [r11]
    vfmadd231sd xmm0, xmm2, xmm3
    
    add r10, 8
    add r11, 8
    dec rcx
    jnz .f64_remainder

.f64_done:
    vzeroupper
    pop rbp
    ret

; ============================================================================
; EUCLIDEAN DISTANCE SQUARED - Float32
; ============================================================================
; Computes sum((a[i] - b[i])^2) without the sqrt
; Input:  ARG1 = A, ARG2 = B, ARG3 = count
; Output: XMM0 = squared distance (as double)
; ============================================================================
global wu_asm_dist_sq_f32
wu_asm_dist_sq_f32:
    push rbp
    mov rbp, rsp
    
    mov r10, ARG1
    mov r11, ARG2
    mov rcx, ARG3
    
    vxorps ymm0, ymm0, ymm0
    
    cmp rcx, 8
    jl .dist_small

.dist_loop:
    vmovups ymm1, [r10]
    vmovups ymm2, [r11]
    vsubps ymm3, ymm1, ymm2    ; diff = a - b
    vfmadd231ps ymm0, ymm3, ymm3  ; sum += diff * diff
    
    add r10, 32
    add r11, 32
    sub rcx, 8
    
    cmp rcx, 8
    jge .dist_loop

.dist_small:
    ; Horizontal sum
    call wu_asm_hsum_f32
    
    ; Scalar remainder
    test rcx, rcx
    jz .dist_done
    
.dist_remainder:
    vmovss xmm1, [r10]
    vmovss xmm2, [r11]
    vsubss xmm3, xmm1, xmm2
    vfmadd231ss xmm0, xmm3, xmm3
    
    add r10, 4
    add r11, 4
    dec rcx
    jnz .dist_remainder

.dist_done:
    vcvtss2sd xmm0, xmm0, xmm0
    vzeroupper
    pop rbp
    ret


SECTION_RODATA

; Constants for future use
align 32
const_one_f32: times 8 dd 1.0
const_zero_f32: times 8 dd 0.0
const_half_f32: times 8 dd 0.5
