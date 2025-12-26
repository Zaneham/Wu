; Wu Forensics - H.264 QP Boundary Detection
; x86-64 AVX2 implementation
;
; Detects sharp quantisation parameter boundaries that may indicate
; video splicing or compositing from different sources.
;
; Build (Windows): nasm -f win64 -o h264_qp_forensic.obj h264_qp_forensic.asm
; Build (Linux):   nasm -f elf64 -o h264_qp_forensic.o h264_qp_forensic.asm
; Build (macOS):   nasm -f macho64 -o h264_qp_forensic.o h264_qp_forensic.asm

%ifidn __OUTPUT_FORMAT__, win64
    %define ARG1 rcx
    %define ARG2 rdx
    %define ARG3 r8
    %define ARG4 r9
    %define ARG1d ecx
    %define ARG2d edx
    %define ARG3d r8d
    %define ARG4d r9d
%else
    %define ARG1 rdi
    %define ARG2 rsi
    %define ARG3 rdx
    %define ARG4 rcx
    %define ARG1d edi
    %define ARG2d esi
    %define ARG3d edx
    %define ARG4d ecx
%endif

section .data align=32

; Shuffle masks for horizontal boundary detection
wu_qp_shuf_left:  db 0,1,2,3, 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, 24,25,26,27
wu_qp_shuf_right: db 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31, 28,29,30,31

section .text

; ============================================================================
; QP BOUNDARY SCAN - Horizontal
; ============================================================================
; Scans a row of QP values for sharp horizontal boundaries.
; Returns count of boundaries exceeding threshold.
;
; Input:  ARG1 = qp_row      (int8_t*, width_mbs QP values)
;         ARG2 = width_mbs   (int, number of macroblocks in row)
;         ARG3 = threshold   (int, minimum delta to flag, typically 8-12)
;         ARG4 = boundaries  (int32_t*, output: [mb_x, delta] pairs)
;         [stack] = max_out  (int, maximum output pairs)
; Output: EAX = number of boundaries found
;
; Algorithm:
;   For each adjacent pair (qp[i], qp[i+1]):
;     delta = |qp[i] - qp[i+1]|
;     if delta >= threshold: record (i, delta)
; ============================================================================
global wu_asm_qp_scan_horizontal
wu_asm_qp_scan_horizontal:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    ; Arguments
    mov r10, ARG1               ; qp_row
    mov r11d, ARG2d             ; width_mbs
    mov r12d, ARG3d             ; threshold
    mov r13, ARG4               ; boundaries output

%ifidn __OUTPUT_FORMAT__, win64
    mov r14d, [rbp + 48]        ; max_out
%else
    mov r14d, r8d               ; max_out (5th arg in r8 for SysV)
%endif

    xor r15d, r15d              ; count = 0

    ; Need at least 2 MBs for a boundary
    cmp r11d, 2
    jl .scan_done

    ; Process in blocks of 32 bytes (32 QP values)
    mov eax, r11d
    sub eax, 1                  ; We check pairs, so width-1 comparisons
    mov ecx, eax
    shr ecx, 5                  ; Number of full 32-byte blocks
    test ecx, ecx
    jz .scan_remainder

    ; Broadcast threshold to ymm for comparison
    movd xmm0, r12d
    vpbroadcastb ymm0, xmm0     ; ymm0 = threshold in all bytes

.scan_loop_32:
    ; Load 32 QP values starting at current position
    vmovdqu ymm1, [r10]         ; qp[i..i+31]
    vmovdqu ymm2, [r10 + 1]     ; qp[i+1..i+32]

    ; Compute absolute difference
    ; |a - b| = max(a-b, b-a) for unsigned, but QP is 0-51, so use saturating sub
    vpsubusb ymm3, ymm1, ymm2   ; max(qp[i] - qp[i+1], 0)
    vpsubusb ymm4, ymm2, ymm1   ; max(qp[i+1] - qp[i], 0)
    vpor ymm3, ymm3, ymm4       ; |delta|

    ; Compare with threshold
    vpcmpgtb ymm4, ymm3, ymm0   ; delta > threshold (signed compare, but values are small)

    ; Extract mask
    vpmovmskb eax, ymm4

    ; Process set bits
    test eax, eax
    jz .scan_next_32

    ; For each set bit, record the boundary
    xor ebx, ebx                ; bit index
.extract_bits_32:
    test eax, 1
    jz .next_bit_32

    ; Check output limit
    cmp r15d, r14d
    jge .scan_done

    ; Calculate absolute MB position
    mov ecx, ebx
    ; r10 offset from original qp_row determines position
    ; We need to track base position - for now, use relative

    ; Store [mb_x, delta]
    mov rdx, r15
    shl rdx, 3                  ; * 8 (two int32s)
    add rdx, r13

    ; Get actual delta value
    movzx ecx, byte [r10 + rbx]
    movzx edx, byte [r10 + rbx + 1]
    sub ecx, edx
    ; Absolute value
    mov eax, ecx
    neg ecx
    cmovl ecx, eax

    ; Store position and delta
    mov rdx, r15
    shl rdx, 3
    add rdx, r13
    mov [rdx], ebx              ; mb_x (relative to current chunk)
    mov [rdx + 4], ecx          ; delta

    inc r15d

.next_bit_32:
    shr eax, 1
    inc ebx
    cmp ebx, 32
    jl .extract_bits_32

.scan_next_32:
    add r10, 32
    dec ecx
    jnz .scan_loop_32

.scan_remainder:
    ; Handle remaining 1-31 pairs with scalar code
    mov eax, r11d
    sub eax, 1
    and eax, 31                 ; Remaining count
    test eax, eax
    jz .scan_done

.scan_scalar:
    ; Check output limit
    cmp r15d, r14d
    jge .scan_done

    movzx ecx, byte [r10]       ; qp[i]
    movzx edx, byte [r10 + 1]   ; qp[i+1]
    sub ecx, edx
    ; Absolute value
    mov ebx, ecx
    neg ecx
    cmovl ecx, ebx

    ; Compare with threshold
    cmp ecx, r12d
    jl .scan_scalar_next

    ; Store boundary
    mov rdx, r15
    shl rdx, 3
    add rdx, r13
    ; Position calculation would need base offset tracking
    mov dword [rdx], 0          ; Placeholder position
    mov [rdx + 4], ecx          ; delta
    inc r15d

.scan_scalar_next:
    inc r10
    dec eax
    jnz .scan_scalar

.scan_done:
    mov eax, r15d               ; Return count

    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


; ============================================================================
; QP FRAME STATISTICS
; ============================================================================
; Computes QP statistics for an entire frame.
; Returns sum, min, max for histogram/analysis.
;
; Input:  ARG1 = qp_map       (int8_t*, height_mbs * width_mbs)
;         ARG2 = width_mbs    (int)
;         ARG3 = height_mbs   (int)
;         ARG4 = stats_out    (int32_t[4]: [sum, min, max, count])
; Output: None (results in stats_out)
; ============================================================================
global wu_asm_qp_frame_stats
wu_asm_qp_frame_stats:
    push rbp
    mov rbp, rsp
    push rbx
    push r12

    mov r10, ARG1               ; qp_map
    mov r11d, ARG2d             ; width_mbs
    mov eax, ARG3d              ; height_mbs
    mov r12, ARG4               ; stats_out

    ; Total count
    imul eax, r11d              ; total_mbs = width * height
    mov ecx, eax

    ; Initialise accumulators
    xor eax, eax                ; sum = 0
    mov ebx, 255                ; min = 255
    xor edx, edx                ; max = 0

    ; Process 32 bytes at a time
    mov r8d, ecx
    shr r8d, 5                  ; num 32-byte blocks
    test r8d, r8d
    jz .stats_remainder

    ; Initialise vector accumulators
    vpxor ymm0, ymm0, ymm0      ; sum accumulator (will use horizontal add at end)
    vpcmpeqb ymm1, ymm1, ymm1   ; min = 0xFF (all 1s)
    vpxor ymm2, ymm2, ymm2      ; max = 0

.stats_loop_32:
    vmovdqu ymm3, [r10]

    ; Update min/max
    vpminub ymm1, ymm1, ymm3    ; min
    vpmaxub ymm2, ymm2, ymm3    ; max

    ; Accumulate sum (need to expand to 16-bit to avoid overflow)
    vpmovzxbw ymm4, xmm3        ; Low 16 bytes -> 16 words
    vextracti128 xmm5, ymm3, 1
    vpmovzxbw ymm5, xmm5        ; High 16 bytes -> 16 words
    vpaddw ymm0, ymm0, ymm4
    vpaddw ymm0, ymm0, ymm5

    add r10, 32
    dec r8d
    jnz .stats_loop_32

    ; Reduce min
    vextracti128 xmm3, ymm1, 1
    vpminub xmm1, xmm1, xmm3
    vpsrldq xmm3, xmm1, 8
    vpminub xmm1, xmm1, xmm3
    vpsrldq xmm3, xmm1, 4
    vpminub xmm1, xmm1, xmm3
    vpsrldq xmm3, xmm1, 2
    vpminub xmm1, xmm1, xmm3
    vpsrldq xmm3, xmm1, 1
    vpminub xmm1, xmm1, xmm3
    vpextrb ebx, xmm1, 0

    ; Reduce max
    vextracti128 xmm3, ymm2, 1
    vpmaxub xmm2, xmm2, xmm3
    vpsrldq xmm3, xmm2, 8
    vpmaxub xmm2, xmm2, xmm3
    vpsrldq xmm3, xmm2, 4
    vpmaxub xmm2, xmm2, xmm3
    vpsrldq xmm3, xmm2, 2
    vpmaxub xmm2, xmm2, xmm3
    vpsrldq xmm3, xmm2, 1
    vpmaxub xmm2, xmm2, xmm3
    vpextrb edx, xmm2, 0

    ; Reduce sum (ymm0 has 16-bit words)
    vextracti128 xmm3, ymm0, 1
    vpaddw xmm0, xmm0, xmm3
    vphaddw xmm0, xmm0, xmm0
    vphaddw xmm0, xmm0, xmm0
    vphaddw xmm0, xmm0, xmm0
    vpextrw eax, xmm0, 0

.stats_remainder:
    ; Handle remaining bytes
    and ecx, 31
    test ecx, ecx
    jz .stats_done

.stats_scalar:
    movzx r8d, byte [r10]
    add eax, r8d
    cmp r8d, ebx
    cmovl ebx, r8d
    cmp r8d, edx
    cmovg edx, r8d
    inc r10
    dec ecx
    jnz .stats_scalar

.stats_done:
    ; Store results
    mov [r12], eax              ; sum
    mov [r12 + 4], ebx          ; min
    mov [r12 + 8], edx          ; max
    mov eax, ARG2d
    imul eax, ARG3d
    mov [r12 + 12], eax         ; count

    vzeroupper
    pop r12
    pop rbx
    pop rbp
    ret


; ============================================================================
; QP HISTOGRAM
; ============================================================================
; Builds histogram of QP values (52 bins, 0-51).
;
; Input:  ARG1 = qp_map       (int8_t*, height_mbs * width_mbs)
;         ARG2 = count        (int, total macroblocks)
;         ARG3 = histogram    (int32_t[52], output)
; Output: None (results in histogram)
; ============================================================================
global wu_asm_qp_histogram
wu_asm_qp_histogram:
    push rbp
    mov rbp, rsp
    push rbx
    push r12

    mov r10, ARG1               ; qp_map
    mov r11d, ARG2d             ; count
    mov r12, ARG3               ; histogram

    ; Clear histogram (52 * 4 = 208 bytes)
    xor eax, eax
    mov ecx, 52
    mov rdi, r12
    rep stosd

    ; Count each QP value
    mov ecx, r11d
    test ecx, ecx
    jz .hist_done

.hist_loop:
    movzx eax, byte [r10]
    cmp eax, 51
    ja .hist_skip               ; Skip invalid QP

    inc dword [r12 + rax*4]

.hist_skip:
    inc r10
    dec ecx
    jnz .hist_loop

.hist_done:
    pop r12
    pop rbx
    pop rbp
    ret


; ============================================================================
; QP DISCONTINUITY DETECTION - Frame Level
; ============================================================================
; Detects frames with large QP changes from previous frame.
; Used for splice detection at frame boundaries.
;
; Input:  ARG1 = avg_qp_history   (float*, n_frames)
;         ARG2 = n_frames         (int)
;         ARG3 = threshold        (float, typically 10-15)
;         ARG4 = discontinuities  (int32_t*, output: [frame_idx, ...])
;         [stack] = max_out       (int)
; Output: EAX = number of discontinuities found
; ============================================================================
global wu_asm_qp_frame_discontinuities
wu_asm_qp_frame_discontinuities:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13

    mov r10, ARG1               ; avg_qp_history
    mov r11d, ARG2d             ; n_frames
    mov r12, ARG4               ; discontinuities output

%ifidn __OUTPUT_FORMAT__, win64
    ; On Win64, float arg3 goes in xmm2, int arg5 on stack
    vmovss xmm0, xmm2           ; threshold in xmm2
    mov r13d, [rbp + 48]        ; max_out
%else
    ; SysV: xmm0 has threshold, 5th int arg in ecx
    ; xmm0 already has threshold
    mov r13d, ecx               ; max_out
%endif

    xor ebx, ebx                ; count = 0

    cmp r11d, 2
    jl .frame_disc_done

    mov ecx, 1                  ; Start from frame 1

.frame_disc_loop:
    cmp ecx, r11d
    jge .frame_disc_done

    cmp ebx, r13d
    jge .frame_disc_done

    ; Load qp[i] and qp[i-1]
    lea eax, [ecx - 1]
    vmovss xmm1, [r10 + rax*4]  ; qp[i-1]
    vmovss xmm2, [r10 + rcx*4]  ; qp[i]

    ; Compute |qp[i] - qp[i-1]}
    vsubss xmm3, xmm2, xmm1

    ; Manual abs: max(delta, -delta)
    vxorps xmm4, xmm4, xmm4
    vsubss xmm5, xmm4, xmm3     ; -delta
    vmaxss xmm3, xmm3, xmm5     ; |delta|

    ; Compare with threshold
    vucomiss xmm3, xmm0
    jbe .frame_disc_next

    ; Record discontinuity
    mov eax, ebx
    shl eax, 2
    add rax, r12
    mov [rax], ecx
    inc ebx

.frame_disc_next:
    inc ecx
    jmp .frame_disc_loop

.frame_disc_done:
    mov eax, ebx

    vzeroupper
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
