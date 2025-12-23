; -----------------------------------------------------------------------------
; Wu Forensics - H.264 Motion Compensation (6-tap Interpolation)
;
; Implements the 6-tap filter for half-pixel interpolation:
;   H = (A - 5B + 20C + 20D - 5E + F + 16) >> 5
;
; Optimised for x86-64 using AVX2.
; -----------------------------------------------------------------------------

%include "common.asm"

SECTION_TEXT

global wu_h264_filter_6tap_avx2

; void wu_h264_filter_6tap_avx2(uint8_t* src, int stride, int16_t* dst, int width, int height)
wu_h264_filter_6tap_avx2:
    PUSH_XMM 6
    
%ifidn __OUTPUT_FORMAT__, win64
    mov r10, [rsp + 40 + 6*16] ; height
    mov r11, rcx               ; src
    mov rax, r8                ; dst
    mov rsi, rdx               ; stride
    mov rdi, r9                ; width
%else
    mov r10, r8                ; height
    mov r11, rdi               ; src
    mov rax, rdx               ; dst
    mov rsi, rsi               ; stride (already in rsi)
    mov rdi, rcx               ; width (in rcx for SysV)
%endif

    vpbroadcastw ymm4, [rel .coeffs_20]
    vpbroadcastw ymm5, [rel .coeffs_5]
    vpbroadcastw ymm6, [rel .round]

.row_loop:
    test r10, r10
    jz .done
    
    ; Process 8 pixels at a time
    mov rcx, rdi ; width
    mov r12, r11 ; src_ptr
    mov r13, rax ; dst_ptr
    
.col_loop:
    cmp rcx, 8
    jl .next_row
    
    ; Load 16 bytes starting from r12-2
    ; [A B C D E F G H ...]
    vmovdqu xmm0, [r12 - 2]
    vpmovzxbw ymm0, xmm0    ; ymm0 = P0 P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 P11 P12 P13 P14 P15
    
    ; A: [0..7], B: [1..8], C: [2..9], D: [3..10], E: [4..11], F: [5..12]
    
    ; Extract shifted versions efficiently
    ; (For this version, we'll use slow shifts but correct math)
    vmovdqa ymm1, ymm0      ; P0...
    
    ; Shift ymm0 to get P1, P2...
    ; Actually, vpsrldq is for bytes. For words, we use vpsrldq with 2*n.
    
    vpsrldq ymm1, ymm0, 2   ; ymm1 = P1 P2 P3 ... (B)
    vpsrldq ymm2, ymm0, 4   ; ymm2 = P2 P3 P4 ... (C)
    vpsrldq ymm3, ymm0, 6   ; ymm3 = P3 P4 P5 ... (D)
    
    ; a+f: ymm0 + (P5 from ymm0)
    ; (Instead of many shifts, just do C+D first)
    vpaddw ymm7, ymm2, ymm3  ; C+D
    vpmullw ymm7, ymm7, ymm4 ; (C+D)*20
    
    ; -5*(B+E)
    vpsrldq ymm8, ymm0, 8    ; P4... (E)
    vpaddw ymm8, ymm8, ymm1  ; B+E
    vpmullw ymm8, ymm8, ymm5 ; (B+E)*5
    vpsubw ymm7, ymm7, ymm8  ; (C+D)*20 - (B+E)*5
    
    ; +(A+F)
    vpsrldq ymm9, ymm0, 10   ; P5... (F)
    vpaddw ymm9, ymm9, ymm0  ; A+F
    vpaddw ymm7, ymm7, ymm9  ; Result!
    
    ; Store high-precision sums in dst (as int16)
    ; We don't shift/round here, we let the C/Python layer handle it or do it in vertical pass.
    vmovdqu [r13], xmm7
    vextracti128 [r13 + 16], ymm7, 1
    
    add r12, 8
    add r13, 16
    sub rcx, 8
    jmp .col_loop

.next_row:
    add r11, rsi ; src += stride
    imul rbx, rdi, 2
    add rax, rbx ; dst += width*2
    dec r10
    jmp .row_loop
    
.done:
    POP_XMM 6
    ret

SECTION_RODATA
    align 16
    .coeffs_20: dw 20, 20, 20, 20, 20, 20, 20, 20
    .coeffs_5:  dw 5, 5, 5, 5, 5, 5, 5, 5
    .round:     dw 16, 16, 16, 16, 16, 16, 16, 16
