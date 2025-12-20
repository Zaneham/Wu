; Wu Forensics - Common SIMD Utilities
; x86-64 AVX2/FMA implementation
;
; This file contains shared utility functions compiled once.

%include "common.asm"

SECTION_TEXT

; ============================================================================
; HORIZONTAL SUM - Float (8 floats in YMM -> single float)
; ============================================================================
global wu_asm_hsum_f32
wu_asm_hsum_f32:
    vperm2f128 ymm1, ymm0, ymm0, 0x01
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    movshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vzeroupper
    ret

; ============================================================================
; HORIZONTAL SUM - Double (4 doubles in YMM -> single double)
; ============================================================================
global wu_asm_hsum_f64
wu_asm_hsum_f64:
    vperm2f128 ymm1, ymm0, ymm0, 0x01
    vaddpd ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0
    vzeroupper
    ret

; ============================================================================
; HORIZONTAL MAX - Float (8 floats in YMM -> single max float)
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
; DOT PRODUCT - Float32
; ============================================================================
global wu_asm_dot_f32
wu_asm_dot_f32:
    push rbp
    mov rbp, rsp
    push rbx
    mov r10, ARG1
    mov r11, ARG2
    mov rcx, ARG3
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    cmp rcx, 32
    jl .small_loop
    prefetchnta [r10 + 256]
    prefetchnta [r11 + 256]
.unrolled_loop:
    vmovups ymm4, [r10]
    vmovups ymm5, [r11]
    vfmadd231ps ymm0, ymm4, ymm5
    vmovups ymm6, [r10 + 32]
    vmovups ymm7, [r11 + 32]
    vfmadd231ps ymm1, ymm6, ymm7
    vmovups ymm4, [r10 + 64]
    vmovups ymm5, [r11 + 64]
    vfmadd231ps ymm2, ymm4, ymm5
    vmovups ymm6, [r10 + 96]
    vmovups ymm7, [r11 + 96]
    vfmadd231ps ymm3, ymm6, ymm7
    prefetchnta [r10 + 384]
    prefetchnta [r11 + 384]
    add r10, 128
    add r11, 128
    sub rcx, 32
    cmp rcx, 32
    jge .unrolled_loop
.small_loop:
    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm0, ymm0, ymm2
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
    call wu_asm_hsum_f32
    test rcx, rcx
    jz .done
    vxorps xmm1, xmm1, xmm1
.scalar_remainder:
    vmovss xmm2, [r10]
    vmovss xmm3, [r11]
    vfmadd231ss xmm1, xmm2, xmm3
    add r10, 4
    add r11, 4
    dec rcx
    jnz .scalar_remainder
    vaddss xmm0, xmm0, xmm1
.done:
    vcvtss2sd xmm0, xmm0, xmm0
    vzeroupper
    pop rbx
    pop rbp
    ret

; ============================================================================
; DOT PRODUCT - Float64
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
    vsubps ymm3, ymm1, ymm2
    vfmadd231ps ymm0, ymm3, ymm3
    add r10, 32
    add r11, 32
    sub rcx, 8
    cmp rcx, 8
    jge .dist_loop
.dist_small:
    call wu_asm_hsum_f32
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
