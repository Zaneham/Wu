; -----------------------------------------------------------------------------
; Wu Forensics - H.264 Integer Inverse Transform (4x4)
;
; ZH:
; Implements the inverse transform specified in ISO/IEC 14496-10 (H.264).
; Optimised for x86-64 using AVX2.
;
; The 4x4 inverse transform is:
;   W = T * H * T'
; where H is the 4x4 matrix of scaled coefficients.
; -----------------------------------------------------------------------------

%include "common.asm"

section .text

global wu_h264_idct_4x4_avx2

; void wu_h264_idct_4x4_avx2(int16_t* block)
; RCX: pointer to 16-element int16_t block (4x4)
wu_h264_idct_4x4_avx2:
    PUSH_XMM 6
    
    ; Load 4 rows of 4 shorts into YMM
    vmovdqu xmm0, [rcx]         ; Rows 0 and 1
    vmovdqu xmm1, [rcx + 16]    ; Rows 2 and 3

    ; Note: For a single 4x4 block, we use XMM. AVX2 could do multiple blocks.
    ; This implementation focuses on the vertical then horizontal pass.

    ; --- Vertical Pass ---
    ; Row 0: xmm0 [0..3], Row 1: xmm0 [4..7], Row 2: xmm1 [0..3], Row 3: xmm1 [4..7]
    
    ; E0 = Row 0 + Row 2
    ; E1 = Row 0 - Row 2
    ; E2 = (Row 1 >> 1) - Row 3
    ; E3 = Row 1 + (Row 3 >> 1)
    
    vmovdqa xmm2, xmm0          ; Row 0 and 1
    vpsraw xmm3, xmm0, 1        ; Row 1 >> 1 (actually Row 0 and 1 shifted)
    
    ; We need to unpack rows to work on them separately or use horizontal ops
    ; but classic H.264 IDCT is easier with row-wise vectors if we have 4 blocks.
    ; For a single block, we'll use a more scalar-ish approach but with SIMD ops.

    ; Extract rows
    vmovq xmm2, xmm0            ; Row 0
    vpextrq r8, xmm0, 1
    vmovq xmm3, r8              ; Row 1
    vmovq xmm4, xmm1            ; Row 2
    vpextrq r9, xmm1, 1
    vmovq xmm5, r9              ; Row 3

    ; Row 0: xmm2, Row 1: xmm3, Row 2: xmm4, Row 3: xmm5
    
    ; a = r0 + r2
    ; b = r0 - r2
    ; c = (r1 >> 1) - r3
    ; d = r1 + (r3 >> 1)
    
    vpaddsw xmm6, xmm2, xmm4     ; a = r0 + r2
    vpsubsw xmm7, xmm2, xmm4    ; b = r0 - r2
    
    vpsraw xmm8, xmm3, 1        ; r1 >> 1
    vpsubsw xmm8, xmm8, xmm5    ; c = (r1 >> 1) - r3
    
    vpsraw xmm9, xmm5, 1        ; r3 >> 1
    vpaddsw xmm9, xmm9, xmm3     ; d = r1 + (r3 >> 1)
    
    ; r0' = a + d
    ; r1' = b + c
    ; r2' = b - c
    ; r3' = a - d

    vpaddsw xmm2, xmm6, xmm9     ; r0'
    vpaddsw xmm3, xmm7, xmm8     ; r1'
    vpsubsw xmm4, xmm7, xmm8    ; r2'
    vpsubsw xmm5, xmm6, xmm9    ; r3'

    ; --- Transpose 4x4 ---
    ; [ r0 ]  -> [ c0 ]
    ; [ r1 ]     [ c1 ]
    ; [ r2 ]     [ c2 ]
    ; [ r3 ]     [ c3 ]
    
    ; Initial state (low 64 bits):
    ; xmm2: r0.0 r0.1 r0.2 r0.3
    ; xmm3: r1.0 r1.1 r1.2 r1.3
    ; xmm4: r2.0 r2.1 r2.2 r2.3
    ; xmm5: r3.0 r3.1 r3.2 r3.3

    vpunpcklwd xmm6, xmm2, xmm3 ; r0.0 r1.0 r0.1 r1.1 r0.2 r1.2 r0.3 r1.3
    vpunpcklwd xmm7, xmm4, xmm5 ; r2.0 r3.0 r2.1 r3.1 r2.2 r3.2 r2.3 r3.3
    
    vpunpckldq xmm8, xmm6, xmm7 ; c0: r0.0 r1.0 r2.0 r3.0 | c1_parts...
    vpunpckhdq xmm9, xmm6, xmm7 ; c2: r0.2 r1.2 r2.2 r3.2 | c3_parts...
    
    ; Extract c1 and c3
    vpshufd xmm10, xmm8, 0x0E    ; Move c1 to low 64 bits
    vpshufd xmm11, xmm9, 0x0E    ; Move c3 to low 64 bits
    
    ; Now:
    ; xmm8:  c0 (low 64 bits)
    ; xmm10: c1 (low 64 bits)
    ; xmm9:  c2 (low 64 bits)
    ; xmm11: c3 (low 64 bits)

    vmovdqa xmm2, xmm8
    vmovdqa xmm3, xmm10
    vmovdqa xmm4, xmm9
    vmovdqa xmm5, xmm11

    ; --- Horizontal Pass ---
    ; Repeat the same logic on the transposed rows
    
    vpaddsw xmm6, xmm2, xmm4     ; a = c0 + c2
    vpsubsw xmm7, xmm2, xmm4    ; b = c0 - c2
    
    vpsraw xmm8, xmm3, 1        ; c1 >> 1
    vpsubsw xmm8, xmm8, xmm5    ; c = (c1 >> 1) - c3
    
    vpsraw xmm9, xmm5, 1        ; c3 >> 1
    vpaddsw xmm9, xmm9, xmm3     ; d = c1 + (c3 >> 1)
    
    vpaddsw xmm2, xmm6, xmm9     ; r0''
    vpaddsw xmm3, xmm7, xmm8     ; r1''
    vpsubsw xmm4, xmm7, xmm8    ; r2''
    vpsubsw xmm5, xmm6, xmm9    ; r3''

    ; Final normalisation (divide by 64 and add 32 for rounding)
    ; In H.264 residual IDCT, the scaling is often combined with quantisation.
    ; This is the core transform. We add rounding and shift.
    
    vmovdqa xmm6, [rel .round]
    
    vpaddsw xmm2, xmm2, xmm6
    vpaddsw xmm3, xmm3, xmm6
    vpaddsw xmm4, xmm4, xmm6
    vpaddsw xmm5, xmm5, xmm6
    
    vpsraw xmm2, xmm2, 6
    vpsraw xmm3, xmm3, 6
    vpsraw xmm4, xmm4, 6
    vpsraw xmm5, xmm5, 6

    ; Store back to memory
    vmovq [rcx], xmm2
    vmovq [rcx + 8], xmm3
    vmovq [rcx + 16], xmm4
    vmovq [rcx + 24], xmm5

    POP_XMM 6
    ret

section .rodata
    align 16
    .round: times 8 dw 32
