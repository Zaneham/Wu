; Wu Forensics - H.264 Motion Vector Field Analysis
; x86-64 AVX2 implementation
;
; Detects motion vector discontinuities that may indicate
; video splicing, object insertion, or tampering.
;
; Build (Windows): nasm -f win64 -o h264_mv_forensic.obj h264_mv_forensic.asm
; Build (Linux):   nasm -f elf64 -o h264_mv_forensic.o h264_mv_forensic.asm
; Build (macOS):   nasm -f macho64 -o h264_mv_forensic.o h264_mv_forensic.asm

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

; Constants for MV analysis
wu_mv_sign_mask:    times 8 dd 0x80000000
wu_mv_abs_mask:     times 8 dd 0x7FFFFFFF

section .text

; ============================================================================
; MV SPATIAL DISCONTINUITY DETECTION - Horizontal
; ============================================================================
; Scans a row of motion vectors for sharp horizontal discontinuities.
; Motion vectors are stored as packed (mvx, mvy) int16_t pairs.
;
; Input:  ARG1 = mv_row        (int16_t*, width_mbs * 2 values: [mvx0,mvy0,mvx1,mvy1,...])
;         ARG2 = width_mbs     (int, number of macroblocks in row)
;         ARG3 = threshold_sq  (int, squared magnitude threshold, typically 1024 = 32^2)
;         ARG4 = discontinuities (int32_t*, output: [mb_x, magnitude] pairs)
;         [stack] = max_out    (int, maximum output pairs)
; Output: EAX = number of discontinuities found
;
; Algorithm:
;   For each adjacent pair of MVs:
;     delta_x = mv[i].x - mv[i+1].x
;     delta_y = mv[i].y - mv[i+1].y
;     magnitude_sq = delta_x^2 + delta_y^2
;     if magnitude_sq >= threshold_sq: record discontinuity
; ============================================================================
global wu_asm_mv_scan_horizontal
wu_asm_mv_scan_horizontal:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r10, ARG1               ; mv_row
    mov r11d, ARG2d             ; width_mbs
    mov r12d, ARG3d             ; threshold_sq
    mov r13, ARG4               ; discontinuities output

%ifidn __OUTPUT_FORMAT__, win64
    mov r14d, [rbp + 48]        ; max_out
%else
    mov r14d, r8d               ; max_out (5th arg in r8 for SysV)
%endif

    xor r15d, r15d              ; count = 0

    ; Need at least 2 MBs for a discontinuity
    cmp r11d, 2
    jl .scan_done

    ; Broadcast threshold for comparison
    movd xmm0, r12d
    vpbroadcastd ymm0, xmm0     ; ymm0 = threshold_sq in all dwords

    ; Process pairs: each MB has 4 bytes (2x int16)
    ; We process 8 MV pairs at a time (8 MVs = 32 bytes)
    mov eax, r11d
    sub eax, 1                  ; width-1 comparisons
    mov ecx, eax
    shr ecx, 3                  ; Number of 8-pair blocks
    test ecx, ecx
    jz .scan_remainder

.scan_loop_8:
    ; Load 8 MVs starting at current position (each MV is 4 bytes)
    vmovdqu ymm1, [r10]         ; mv[i..i+7]
    vmovdqu ymm2, [r10 + 4]     ; mv[i+1..i+8] (shifted by one MV)

    ; Unpack to separate x and y components
    ; MVs are packed as [x0,y0,x1,y1,...] as int16
    ; We need to compute (x0-x1)^2 + (y0-y1)^2

    ; Subtract packed int16 values
    vpsubw ymm3, ymm1, ymm2     ; [dx0,dy0,dx1,dy1,...]

    ; Square each component (need to handle as signed)
    ; Convert to int32 for squaring to avoid overflow
    vpmovsxwd ymm4, xmm3        ; Sign-extend low 8 int16 to int32
    vextracti128 xmm5, ymm3, 1
    vpmovsxwd ymm5, xmm5        ; Sign-extend high 8 int16 to int32

    ; Square
    vpmulld ymm4, ymm4, ymm4    ; dx^2 or dy^2 for first 8
    vpmulld ymm5, ymm5, ymm5    ; dx^2 or dy^2 for second 8

    ; Now we have [dx0^2, dy0^2, dx1^2, dy1^2, ...] in ymm4/ymm5
    ; Need to add adjacent pairs to get magnitude^2

    ; Horizontal add adjacent pairs
    vphaddd ymm4, ymm4, ymm5    ; Pairs summed

    ; Compare with threshold
    vpcmpgtd ymm6, ymm4, ymm0   ; magnitude_sq > threshold

    ; Extract mask (we have 8 results now, one per MV pair)
    vmovmskps eax, ymm6

    test eax, eax
    jz .scan_next_8

    ; Process set bits
    xor ebx, ebx
.extract_bits_8:
    test eax, 1
    jz .next_bit_8

    cmp r15d, r14d
    jge .scan_done

    ; Store discontinuity
    mov rdx, r15
    shl rdx, 3
    add rdx, r13
    mov [rdx], ebx              ; mb_x (relative)

    ; Extract magnitude from ymm4
    vextractps [rdx + 4], xmm4, 0  ; Store magnitude_sq

    inc r15d

.next_bit_8:
    ; Rotate ymm4 to get next value
    vpsrldq xmm4, xmm4, 4
    shr eax, 1
    inc ebx
    cmp ebx, 8
    jl .extract_bits_8

.scan_next_8:
    add r10, 32                 ; Advance 8 MVs
    dec ecx
    jnz .scan_loop_8

.scan_remainder:
    ; Handle remaining 1-7 pairs with scalar code
    mov eax, r11d
    sub eax, 1
    and eax, 7
    test eax, eax
    jz .scan_done

.scan_scalar:
    cmp r15d, r14d
    jge .scan_done

    ; Load two adjacent MVs
    movsx ecx, word [r10]       ; mv[i].x
    movsx edx, word [r10 + 2]   ; mv[i].y
    movsx ebx, word [r10 + 4]   ; mv[i+1].x
    movsx r8d, word [r10 + 6]   ; mv[i+1].y

    ; Compute deltas
    sub ecx, ebx                ; dx
    sub edx, r8d                ; dy

    ; Compute magnitude squared
    imul ecx, ecx               ; dx^2
    imul edx, edx               ; dy^2
    add ecx, edx                ; magnitude_sq

    ; Compare with threshold
    cmp ecx, r12d
    jl .scan_scalar_next

    ; Store discontinuity
    mov rdx, r15
    shl rdx, 3
    add rdx, r13
    mov dword [rdx], 0          ; Position placeholder
    mov [rdx + 4], ecx          ; magnitude_sq
    inc r15d

.scan_scalar_next:
    add r10, 4                  ; Next MV
    dec eax
    jnz .scan_scalar

.scan_done:
    mov eax, r15d

    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


; ============================================================================
; MV FIELD STATISTICS
; ============================================================================
; Computes motion vector field statistics for forensic analysis.
; Returns sum of magnitudes, max magnitude, and outlier count.
;
; Input:  ARG1 = mv_field      (int16_t*, height_mbs * width_mbs * 2)
;         ARG2 = width_mbs     (int)
;         ARG3 = height_mbs    (int)
;         ARG4 = stats_out     (int32_t[4]: [sum_mag, max_mag, outlier_count, total])
;         [stack] = outlier_threshold (int, magnitude threshold for outlier)
; Output: None (results in stats_out)
; ============================================================================
global wu_asm_mv_field_stats
wu_asm_mv_field_stats:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14

    mov r10, ARG1               ; mv_field
    mov r11d, ARG2d             ; width_mbs
    mov eax, ARG3d              ; height_mbs
    mov r12, ARG4               ; stats_out

%ifidn __OUTPUT_FORMAT__, win64
    mov r13d, [rbp + 48]        ; outlier_threshold
%else
    mov r13d, r8d               ; outlier_threshold (5th arg)
%endif

    ; Total count
    imul eax, r11d              ; total_mbs = width * height
    mov r14d, eax               ; Save total

    ; Initialise accumulators
    xor eax, eax                ; sum_mag = 0
    xor ebx, ebx                ; max_mag = 0
    xor ecx, ecx                ; outlier_count = 0

    ; Process each MV
    mov r8d, r14d
    test r8d, r8d
    jz .stats_done

.stats_loop:
    ; Load MV (x, y)
    movsx r9d, word [r10]       ; mvx
    movsx r11d, word [r10 + 2]  ; mvy

    ; Compute magnitude squared
    imul r9d, r9d
    imul r11d, r11d
    add r9d, r11d               ; magnitude_sq

    ; Update sum (use magnitude, not squared, for interpretability)
    ; Approximate sqrt via shift for speed: sqrt(x) ≈ x >> (log2(x)/2)
    ; For simplicity, just accumulate squared values
    add eax, r9d

    ; Update max
    cmp r9d, ebx
    cmovg ebx, r9d

    ; Check outlier
    cmp r9d, r13d
    jl .not_outlier
    inc ecx
.not_outlier:

    add r10, 4                  ; Next MV
    dec r8d
    jnz .stats_loop

.stats_done:
    mov [r12], eax              ; sum_magnitude_sq
    mov [r12 + 4], ebx          ; max_magnitude_sq
    mov [r12 + 8], ecx          ; outlier_count
    mov [r12 + 12], r14d        ; total

    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


; ============================================================================
; MV FIELD COHERENCE SCORE
; ============================================================================
; Computes a coherence score for the motion vector field.
; Coherent fields (natural video) have smooth MV gradients.
; Spliced/manipulated regions often have discontinuous MVs.
;
; Input:  ARG1 = mv_field      (int16_t*, height_mbs * width_mbs * 2)
;         ARG2 = width_mbs     (int)
;         ARG3 = height_mbs    (int)
; Output: EAX = coherence score (0-100, higher = more coherent)
;
; Algorithm:
;   For each internal MB, compute gradient to 4 neighbours.
;   Coherence = 100 - (average_gradient / max_expected_gradient * 100)
; ============================================================================
global wu_asm_mv_coherence_score
wu_asm_mv_coherence_score:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r10, ARG1               ; mv_field
    mov r11d, ARG2d             ; width_mbs
    mov r12d, ARG3d             ; height_mbs

    ; Need at least 3x3 for internal MBs
    cmp r11d, 3
    jl .coherence_zero
    cmp r12d, 3
    jl .coherence_zero

    xor r13d, r13d              ; total_gradient = 0
    xor r14d, r14d              ; count = 0

    ; Row stride in bytes
    mov eax, r11d
    shl eax, 2                  ; width_mbs * 4 bytes per MV
    mov r15d, eax               ; row_stride

    ; Start at (1,1), end at (width-2, height-2)
    mov ecx, 1                  ; y = 1
.coherence_row_loop:
    cmp ecx, r12d
    jge .coherence_calc
    lea eax, [ecx - 1]
    cmp eax, r12d
    jge .coherence_calc

    mov ebx, 1                  ; x = 1
.coherence_col_loop:
    lea eax, [r11d - 1]
    cmp ebx, eax
    jge .coherence_next_row

    ; Calculate offset to current MB
    mov eax, ecx
    imul eax, r15d              ; y * row_stride
    lea edx, [ebx * 4]          ; x * 4
    add eax, edx                ; offset

    ; Load current MV
    movsx r8d, word [r10 + rax]
    movsx r9d, word [r10 + rax + 2]

    ; Load left neighbour
    movsx edx, word [r10 + rax - 4]
    sub edx, r8d
    imul edx, edx
    movsx edi, word [r10 + rax - 2]
    sub edi, r9d
    imul edi, edi
    add edx, edi
    add r13d, edx               ; Add gradient to left

    ; Load right neighbour
    movsx edx, word [r10 + rax + 4]
    sub edx, r8d
    imul edx, edx
    movsx edi, word [r10 + rax + 6]
    sub edi, r9d
    imul edi, edi
    add edx, edi
    add r13d, edx               ; Add gradient to right

    ; Load top neighbour
    mov edi, eax
    sub edi, r15d               ; offset - row_stride
    movsx edx, word [r10 + rdi]
    sub edx, r8d
    imul edx, edx
    movsx esi, word [r10 + rdi + 2]
    sub esi, r9d
    imul esi, esi
    add edx, esi
    add r13d, edx               ; Add gradient to top

    ; Load bottom neighbour
    mov edi, eax
    add edi, r15d               ; offset + row_stride
    movsx edx, word [r10 + rdi]
    sub edx, r8d
    imul edx, edx
    movsx esi, word [r10 + rdi + 2]
    sub esi, r9d
    imul esi, esi
    add edx, esi
    add r13d, edx               ; Add gradient to bottom

    add r14d, 4                 ; count += 4 gradients

    inc ebx
    jmp .coherence_col_loop

.coherence_next_row:
    inc ecx
    jmp .coherence_row_loop

.coherence_calc:
    ; Calculate average and convert to score
    test r14d, r14d
    jz .coherence_zero

    ; average = total / count
    mov eax, r13d
    xor edx, edx
    div r14d                    ; eax = average_gradient

    ; Normalise: assume max expected gradient is 10000 (100 pixels squared)
    ; score = 100 - min(average / 100, 100)
    mov ecx, 100
    xor edx, edx
    div ecx                     ; eax = average / 100

    cmp eax, 100
    jle .coherence_clamp
    mov eax, 100
.coherence_clamp:
    mov ecx, 100
    sub ecx, eax
    mov eax, ecx
    jmp .coherence_done

.coherence_zero:
    mov eax, 100                ; No data = assume coherent

.coherence_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


; ============================================================================
; MV OUTLIER DETECTION
; ============================================================================
; Identifies macroblocks with motion vectors that deviate significantly
; from their local neighbourhood.
;
; Input:  ARG1 = mv_field      (int16_t*, height_mbs * width_mbs * 2)
;         ARG2 = width_mbs     (int)
;         ARG3 = height_mbs    (int)
;         ARG4 = outliers      (int32_t*, output: [mb_x, mb_y, deviation] triples)
;         [stack] = threshold  (int, deviation threshold, typically 2500 = 50^2)
;         [stack+8] = max_out  (int)
; Output: EAX = number of outliers found
; ============================================================================
global wu_asm_mv_detect_outliers
wu_asm_mv_detect_outliers:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16                 ; Local storage

    mov r10, ARG1               ; mv_field
    mov r11d, ARG2d             ; width_mbs
    mov r12d, ARG3d             ; height_mbs
    mov r13, ARG4               ; outliers output

%ifidn __OUTPUT_FORMAT__, win64
    mov r14d, [rbp + 48]        ; threshold
    mov r15d, [rbp + 56]        ; max_out
%else
    mov r14d, r8d               ; threshold
    mov r15d, r9d               ; max_out
%endif

    mov [rbp - 8], r14          ; Store threshold locally
    xor r14d, r14d              ; count = 0

    ; Need at least 3x3
    cmp r11d, 3
    jl .outlier_done
    cmp r12d, 3
    jl .outlier_done

    ; Row stride
    mov eax, r11d
    shl eax, 2
    mov [rbp - 16], eax         ; row_stride

    mov ecx, 1                  ; y = 1
.outlier_row_loop:
    lea eax, [r12d - 1]
    cmp ecx, eax
    jge .outlier_done

    mov ebx, 1                  ; x = 1
.outlier_col_loop:
    lea eax, [r11d - 1]
    cmp ebx, eax
    jge .outlier_next_row

    cmp r14d, r15d
    jge .outlier_done

    ; Calculate offset
    mov eax, [rbp - 16]         ; row_stride
    imul eax, ecx               ; y * row_stride
    lea edx, [ebx * 4]
    add eax, edx                ; offset

    ; Load current MV
    movsx r8d, word [r10 + rax]
    movsx r9d, word [r10 + rax + 2]

    ; Compute average of 4 neighbours
    xor edi, edi                ; sum_x
    xor esi, esi                ; sum_y

    ; Left
    movsx edx, word [r10 + rax - 4]
    add edi, edx
    movsx edx, word [r10 + rax - 2]
    add esi, edx

    ; Right
    movsx edx, word [r10 + rax + 4]
    add edi, edx
    movsx edx, word [r10 + rax + 6]
    add esi, edx

    ; Top
    push rax
    mov edx, [rbp - 16]
    sub eax, edx
    movsx edx, word [r10 + rax]
    add edi, edx
    movsx edx, word [r10 + rax + 2]
    add esi, edx
    pop rax

    ; Bottom
    push rax
    add eax, [rbp - 16]
    movsx edx, word [r10 + rax]
    add edi, edx
    movsx edx, word [r10 + rax + 2]
    add esi, edx
    pop rax

    ; Average (divide by 4)
    sar edi, 2
    sar esi, 2

    ; Deviation from average
    sub r8d, edi
    sub r9d, esi
    imul r8d, r8d
    imul r9d, r9d
    add r8d, r9d                ; deviation_sq

    ; Compare with threshold
    cmp r8d, [rbp - 8]
    jl .outlier_next_col

    ; Store outlier
    mov rax, r14
    imul rax, 12                ; 3 dwords per entry
    add rax, r13
    mov [rax], ebx              ; mb_x
    mov [rax + 4], ecx          ; mb_y
    mov [rax + 8], r8d          ; deviation_sq
    inc r14d

.outlier_next_col:
    inc ebx
    jmp .outlier_col_loop

.outlier_next_row:
    inc ecx
    jmp .outlier_row_loop

.outlier_done:
    mov eax, r14d

    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
