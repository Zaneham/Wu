; Wu Forensics - Block Grid Analysis Assembly
; x86-64 AVX2/FMA implementation
;
; Blockiness computation for JPEG grid detection.
; Tests all 64 offset combinations efficiently.
;
; Build (Windows): nasm -f win64 -o blockgrid.obj blockgrid.asm
; Build (Linux):   nasm -f elf64 -o blockgrid.o blockgrid.asm
; Build (macOS):   nasm -f macho64 -o blockgrid.o blockgrid.asm

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

section .text

; ============================================================================
; COMPUTE BLOCKINESS - Single offset
; ============================================================================
; Computes the blockiness score for a given grid offset.
; Blockiness = average squared difference across block boundaries.
;
; Input:  ARG1 = image     (double*, row-major grayscale)
;         ARG2 = width     (int)
;         ARG3 = height    (int)
;         ARG4 = x_offset  (int, 0-7)
;         [stack] = y_offset (int, 0-7)
;         [stack] = block_size (int, typically 8)
; Output: XMM0 = blockiness score (double)
; ============================================================================
global wu_asm_compute_blockiness
wu_asm_compute_blockiness:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    
    ; Store parameters
    mov [rbp - 8], ARG1         ; image
    mov [rbp - 16], ARG2        ; width (as qword)
    mov [rbp - 24], ARG3        ; height (as qword)
    mov [rbp - 32], ARG4        ; x_offset
    
    ; Load stack args
%ifidn __OUTPUT_FORMAT__, win64
    mov eax, [rbp + 0x30]       ; y_offset
    mov ecx, [rbp + 0x38]       ; block_size
%else
    mov eax, [rbp + 0x10]       ; y_offset
    mov ecx, [rbp + 0x18]       ; block_size
%endif
    mov [rbp - 36], eax
    mov [rbp - 40], ecx
    
    ; Initialize accumulators
    vxorpd ymm0, ymm0, ymm0     ; total_diff accumulator
    xor r12d, r12d              ; count = 0
    
    ; ========== VERTICAL BOUNDARIES ==========
    ; For x = x_offset; x < width - 1; x += block_size
    mov r10d, [rbp - 32]        ; x = x_offset
    
.vert_x_loop:
    mov eax, [rbp - 16]         ; width
    dec eax                     ; width - 1
    cmp r10d, eax
    jge .horiz_start
    
    ; For y = 0; y < height; y++
    xor r11d, r11d              ; y = 0
    
.vert_y_loop:
    cmp r11d, [rbp - 24]        ; y < height?
    jge .vert_next_x
    
    ; Compute address: image[y * width + x]
    mov rax, r11                ; y
    imul rax, [rbp - 16]        ; y * width
    add rax, r10                ; + x
    shl rax, 3                  ; * 8 (sizeof double)
    mov rcx, [rbp - 8]          ; image base
    add rcx, rax                ; &image[y * width + x]
    
    ; Load image[y*width + x] and image[y*width + x + 1]
    vmovsd xmm1, [rcx]          ; pixel at (x, y)
    vmovsd xmm2, [rcx + 8]      ; pixel at (x+1, y)
    
    ; diff = pixel1 - pixel2
    vsubsd xmm3, xmm1, xmm2
    ; diff_sq = diff * diff
    vmulsd xmm3, xmm3, xmm3
    ; Accumulate
    vaddsd xmm0, xmm0, xmm3
    
    inc r12d                    ; count++
    inc r11d                    ; y++
    jmp .vert_y_loop
    
.vert_next_x:
    add r10d, [rbp - 40]        ; x += block_size
    jmp .vert_x_loop
    
.horiz_start:
    ; ========== HORIZONTAL BOUNDARIES ==========
    ; For y = y_offset; y < height - 1; y += block_size
    mov r10d, [rbp - 36]        ; y = y_offset
    
.horiz_y_loop:
    mov eax, [rbp - 24]         ; height
    dec eax                     ; height - 1
    cmp r10d, eax
    jge .compute_average
    
    ; For x = 0; x < width; x++
    xor r11d, r11d              ; x = 0
    
.horiz_x_loop:
    cmp r11d, [rbp - 16]        ; x < width?
    jge .horiz_next_y
    
    ; Compute address: image[y * width + x]
    mov rax, r10                ; y
    imul rax, [rbp - 16]        ; y * width
    add rax, r11                ; + x
    shl rax, 3                  ; * 8
    mov rcx, [rbp - 8]
    add rcx, rax
    
    ; Address of image[(y+1) * width + x]
    mov rax, r10
    inc rax                     ; y + 1
    imul rax, [rbp - 16]
    add rax, r11
    shl rax, 3
    mov rdx, [rbp - 8]
    add rdx, rax
    
    ; Load pixels
    vmovsd xmm1, [rcx]          ; (x, y)
    vmovsd xmm2, [rdx]          ; (x, y+1)
    
    ; diff_sq
    vsubsd xmm3, xmm1, xmm2
    vmulsd xmm3, xmm3, xmm3
    vaddsd xmm0, xmm0, xmm3
    
    inc r12d
    inc r11d
    jmp .horiz_x_loop
    
.horiz_next_y:
    add r10d, [rbp - 40]        ; y += block_size
    jmp .horiz_y_loop
    
.compute_average:
    ; average = total_diff / count
    test r12d, r12d
    jz .return_zero
    
    vcvtsi2sd xmm1, xmm1, r12d
    vdivsd xmm0, xmm0, xmm1
    jmp .done

.return_zero:
    vxorpd xmm0, xmm0, xmm0

.done:
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; COMPUTE ALL OFFSETS - Parallel blockiness for 64 offset combinations
; ============================================================================
; Computes blockiness for all 64 grid offsets (8x8) in parallel.
; Returns the best offset and its score.
;
; Input:  ARG1 = image        (double*, row-major grayscale)
;         ARG2 = width        (int)
;         ARG3 = height       (int)
;         ARG4 = block_size   (int, typically 8)
;         [stack] = scores    (double[64], output, row-major [y][x])
; Output: EAX = best offset index (y * 8 + x)
;         XMM0 = best score (double)
; ============================================================================
global wu_asm_blockiness_all_offsets
wu_asm_blockiness_all_offsets:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    
    ; Store parameters
    mov [rbp - 8], ARG1         ; image
    mov [rbp - 16], ARG2        ; width
    mov [rbp - 24], ARG3        ; height
    mov [rbp - 32], ARG4        ; block_size
    
%ifidn __OUTPUT_FORMAT__, win64
    mov rax, [rbp + 0x30]       ; scores output
%else
    mov rax, [rbp + 0x10]
%endif
    mov [rbp - 40], rax
    
    ; Initialize best tracking
    mov dword [rbp - 44], 0     ; best_idx = 0
    vmovsd xmm15, [rel .max_double]  ; best_score = MAX
    
    ; Loop: y_offset = 0 to 7
    xor r14d, r14d              ; y_offset

.offset_y_loop:
    cmp r14d, 8
    jge .find_best
    
    ; x_offset = 0 to 7
    xor r15d, r15d              ; x_offset

.offset_x_loop:
    cmp r15d, 8
    jge .offset_next_y
    
    ; Call wu_asm_compute_blockiness
    ; Set up arguments
%ifidn __OUTPUT_FORMAT__, win64
    mov rcx, [rbp - 8]          ; image
    mov rdx, [rbp - 16]         ; width
    mov r8, [rbp - 24]          ; height
    mov r9d, r15d               ; x_offset
    mov eax, r14d
    mov [rsp + 0x20], eax       ; y_offset on stack
    mov eax, [rbp - 32]
    mov [rsp + 0x28], eax       ; block_size on stack
%else
    mov rdi, [rbp - 8]
    mov rsi, [rbp - 16]
    mov rdx, [rbp - 24]
    mov ecx, r15d
    mov r8d, r14d
    mov r9d, [rbp - 32]
%endif
    
    ; Save state
    vmovsd [rbp - 56], xmm15
    
    call wu_asm_compute_blockiness
    
    ; Result in xmm0
    vmovsd xmm15, [rbp - 56]
    
    ; Store in scores array: scores[y_offset * 8 + x_offset]
    mov rax, r14
    shl rax, 3                  ; y_offset * 8
    add rax, r15                ; + x_offset
    shl rax, 3                  ; * sizeof(double)
    mov rcx, [rbp - 40]         ; scores base
    vmovsd [rcx + rax], xmm0
    
    ; Track minimum (best score = lowest blockiness at wrong offset)
    vucomisd xmm0, xmm15
    jae .offset_next_x
    
    ; New best
    vmovsd xmm15, xmm0
    mov eax, r14d
    shl eax, 3
    add eax, r15d
    mov [rbp - 44], eax

.offset_next_x:
    inc r15d
    jmp .offset_x_loop

.offset_next_y:
    inc r14d
    jmp .offset_y_loop

.find_best:
    ; Return best index in eax, best score in xmm0
    mov eax, [rbp - 44]
    vmovsd xmm0, xmm15
    
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

section .data align=32

.max_double: dq 1.0e308         ; DBL_MAX approximation
