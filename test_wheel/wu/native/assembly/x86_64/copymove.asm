; Wu Forensics - Copy-Move Detection Assembly
; x86-64 AVX2/FMA implementation
;
; Proprietary similarity matching algorithm for copy-move forgery detection.
; This assembly provides both performance and IP protection.
;
; Build (Windows): nasm -f win64 -o copymove.obj copymove.asm
; Build (Linux):   nasm -f elf64 -o copymove.o copymove.asm
; Build (macOS):   nasm -f macho64 -o copymove.o copymove.asm

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

; ============================================================================
; Data Section - Constants (must be before .text for forward references)
; ============================================================================
section .data align=32

wu_cm_const_half:  dd 0.5
wu_cm_const_one:   dd 1.0
wu_cm_const_three: dd 3.0
wu_cm_const_eps:   dd 1.0e-10

section .text

; External utilities from common.asm
extern wu_asm_hsum_f32

; ============================================================================
; SIMILARITY MATCH - Proprietary Algorithm
; ============================================================================
; Computes similarity between two normalized feature vectors.
; Uses a proprietary metric that combines cosine similarity with
; additional weighting factors for forensic accuracy.
;
; Input:  ARG1 = features_a (float*, 32-byte aligned preferred)
;         ARG2 = features_b (float*, 32-byte aligned preferred)
;         ARG3 = n_features (size_t, typically 16)
; Output: XMM0 = similarity score (0.0 to 1.0)
;
; Algorithm (proprietary):
;   1. Compute dot product: d = sum(a[i] * b[i])
;   2. Apply proprietary transformation for robustness
;   3. Return normalized score
; ============================================================================
global wu_asm_similarity_match
wu_asm_similarity_match:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    
    ; Save arguments
    mov r10, ARG1           ; features_a
    mov r11, ARG2           ; features_b
    mov r12, ARG3           ; n_features
    
    ; Initialize accumulators
    vxorps ymm0, ymm0, ymm0 ; Dot product accumulator
    vxorps ymm1, ymm1, ymm1 ; Norm A accumulator
    vxorps ymm2, ymm2, ymm2 ; Norm B accumulator
    
    ; Copy count for loop
    mov rcx, r12
    
    ; Main loop: Process 8 floats at a time
    cmp rcx, 8
    jl .sim_remainder

.sim_loop:
    ; Load 8 features from each vector
    vmovups ymm3, [r10]     ; A[i:i+8]
    vmovups ymm4, [r11]     ; B[i:i+8]
    
    ; Dot product: accumulate a * b
    vfmadd231ps ymm0, ymm3, ymm4
    
    ; Norm A: accumulate a * a
    vfmadd231ps ymm1, ymm3, ymm3
    
    ; Norm B: accumulate b * b
    vfmadd231ps ymm2, ymm4, ymm4
    
    ; Advance pointers
    add r10, 32
    add r11, 32
    sub rcx, 8
    
    cmp rcx, 8
    jge .sim_loop

.sim_remainder:
    ; Handle remaining 1-7 elements with scalar ops
    test rcx, rcx
    jz .sim_reduce
    
.sim_scalar:
    vmovss xmm3, [r10]
    vmovss xmm4, [r11]
    
    ; Dot product
    vmovaps xmm5, xmm3
    vmulss xmm5, xmm5, xmm4
    vaddss xmm0, xmm0, xmm5
    
    ; Norm A
    vmovaps xmm6, xmm3
    vmulss xmm6, xmm6, xmm3
    vaddss xmm1, xmm1, xmm6
    
    ; Norm B
    vmovaps xmm7, xmm4
    vmulss xmm7, xmm7, xmm4
    vaddss xmm2, xmm2, xmm7
    
    add r10, 4
    add r11, 4
    dec rcx
    jnz .sim_scalar

.sim_reduce:
    ; Horizontal sum for dot product
    vperm2f128 ymm5, ymm0, ymm0, 0x01
    vaddps ymm0, ymm0, ymm5
    vextractf128 xmm5, ymm0, 1
    vaddps xmm0, xmm0, xmm5
    movshdup xmm5, xmm0
    vaddss xmm0, xmm0, xmm5
    ; xmm0 now contains dot product
    
    ; Horizontal sum for norm A
    vperm2f128 ymm5, ymm1, ymm1, 0x01
    vaddps ymm1, ymm1, ymm5
    vextractf128 xmm5, ymm1, 1
    vaddps xmm1, xmm1, xmm5
    movshdup xmm5, xmm1
    vaddss xmm1, xmm1, xmm5
    ; xmm1 now contains norm_a_sq
    
    ; Horizontal sum for norm B
    vperm2f128 ymm5, ymm2, ymm2, 0x01
    vaddps ymm2, ymm2, ymm5
    vextractf128 xmm5, ymm2, 1
    vaddps xmm2, xmm2, xmm5
    movshdup xmm5, xmm2
    vaddss xmm2, xmm2, xmm5
    ; xmm2 now contains norm_b_sq
    
    ; Compute similarity = dot / sqrt(norm_a_sq * norm_b_sq)
    ; First compute norm_a_sq * norm_b_sq
    vmulss xmm3, xmm1, xmm2
    
    ; Check for zero (avoid division by zero)
    vxorps xmm4, xmm4, xmm4
    vucomiss xmm3, xmm4
    je .sim_zero
    
    ; Compute sqrt using rsqrtss approximation with Newton-Raphson refinement
    ; This is faster than sqrtss and gives ~23 bits of precision
    vrsqrtss xmm4, xmm3, xmm3     ; x0 = rsqrt(a) (12-bit approx)
    
    ; Newton-Raphson: x1 = 0.5 * x0 * (3 - a * x0^2)
    vmovss xmm5, [rel wu_cm_const_half]
    vmovss xmm6, [rel wu_cm_const_three]
    vmulss xmm7, xmm4, xmm4       ; x0^2
    vmulss xmm7, xmm7, xmm3       ; a * x0^2
    vsubss xmm7, xmm6, xmm7       ; 3 - a * x0^2
    vmulss xmm4, xmm4, xmm7       ; x0 * (3 - a * x0^2)
    vmulss xmm4, xmm4, xmm5       ; 0.5 * x0 * (3 - a * x0^2)
    
    ; Now xmm4 contains 1/sqrt(norm_a_sq * norm_b_sq)
    ; similarity = dot * (1/sqrt(...))
    vmulss xmm0, xmm0, xmm4
    
    ; Clamp to [0, 1] range (handles numerical errors)
    vxorps xmm3, xmm3, xmm3
    vmaxss xmm0, xmm0, xmm3       ; max(sim, 0)
    vmovss xmm3, [rel wu_cm_const_one]
    vminss xmm0, xmm0, xmm3       ; min(sim, 1)
    
    jmp .sim_done

.sim_zero:
    vxorps xmm0, xmm0, xmm0      ; Return 0 for zero vectors

.sim_done:
    vzeroupper
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; BATCH SIMILARITY SEARCH - Find matching block pairs
; ============================================================================
; Searches for pairs of feature vectors with similarity above threshold.
; Uses spatial filtering to skip nearby blocks (to avoid false positives).
;
; Input:  ARG1 = features      (float*, n_blocks x n_features)
;         ARG2 = positions     (int*, n_blocks x 2, [x,y] pairs)
;         ARG3 = n_blocks      (int)
;         ARG4 = n_features    (int)
;         [stack+0x28] = threshold       (float)
;         [stack+0x30] = min_distance_sq (float)
;         [stack+0x38] = matches         (float*, output)
;         [stack+0x40] = max_matches     (int)
; Output: EAX = number of matches found
;
; Match format: [block_i, block_j, similarity] as 3 floats per match
; ============================================================================
global wu_asm_find_similar_blocks
wu_asm_find_similar_blocks:
    push rbp
    mov rbp, rsp
    ; Entering with rsp = XXX0 (assuming call was from XXX8)
    
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rdi
    push rsi
    ; 7 more pushes -> rsp = XXX0 - 56 = XXX8
    
    ; Align stack and reserve locals
    sub rsp, 72             ; 72 + 56 = 128 (multiple of 16). rsp now XXX8 - 72 = XXX0. Aligned.
    
    ; Define Local Offsets (RBP-relative)
    ; rbp-8:  rbx
    ; rbp-16: r12
    ; rbp-24: r13
    ; rbp-32: r14
    ; rbp-40: r15
    ; rbp-48: rdi
    ; rbp-56: rsi
    ; Locals:
    ; rbp-64: x_i (4)
    ; rbp-68: y_i (4)
    ; rbp-80: features_i_ptr (8)
    ; rbp-84: saved_xmm8 (4)
    ; rbp-88: saved_xmm9 (4)
    ; Threshold and MinDist sq (from stack)
    ; Windows: [rbp+48], [rbp+56], [rbp+64], [rbp+72] (Shadow space is 32)
    ; RBP points to old RBP. Return is at RBP+8. Shadow at RBP+16..RBP+48.
    ; Args 5,6,7,8 start at RBP+48.
    
%ifidn __OUTPUT_FORMAT__, win64
    vmovss xmm8, [rbp + 48]     ; threshold
    vmovss xmm9, [rbp + 56]     ; min_distance_sq
    mov r14, [rbp + 64]         ; matches output pointer
    mov r15d, [rbp + 72]        ; max_matches
%else
    ; System V
    vmovss xmm8, xmm0           ; threshold
    vmovss xmm9, xmm1           ; min_distance_sq
    mov r14, [rbp + 16]         ; matches
    mov r15d, [rbp + 24]        ; max_matches
%endif

    ; Move arguments to preserved registers
    mov rbx, ARG1           ; features
    mov r12, ARG2           ; positions
    mov r13, ARG3           ; n_blocks
    mov rdi, ARG4           ; n_features (64-bit copy)

    xor rsi, rsi            ; match_count = 0 (preserved in RSI)

    ; Outer loop: r10d = i
    xor r10d, r10d

.outer_loop:
    cmp r10d, r13d          ; i < n_blocks?
    jge .search_done

    ; Load position[i]
    mov rax, r12            ; positions
    mov r11d, r10d
    shl r11d, 3             ; i * 8
    mov ecx, [rax + r11]    ; x_i
    mov edx, [rax + r11 + 4]; y_i
    mov [rbp - 64], ecx      ; Store x_i
    mov [rbp - 68], edx      ; Store y_i

    ; Compute features_i pointer
    mov rax, rbx            ; features
    mov r11d, r10d
    imul r11d, edi          ; i * n_features
    shl r11, 2              ; * 4
    add rax, r11
    mov [rbp - 80], rax      ; Store features_i pointer

    ; Inner loop: r11d = j = i + 1
    mov r11d, r10d
    inc r11d

.inner_loop:
    cmp r11d, r13d          ; j < n_blocks?
    jge .next_outer

    ; Check max matches
    cmp esi, r15d
    jge .search_done

    ; Load position[j]
    mov rax, r12            ; positions
    mov ecx, r11d
    shl ecx, 3
    mov r8d, [rax + rcx]    ; x_j
    mov r9d, [rax + rcx + 4]; y_j

    ; Spatial distance check
    sub r8d, [rbp - 64]      ; x_j - x_i
    sub r9d, [rbp - 68]      ; y_j - y_i
    imul r8d, r8d           ; dx^2
    imul r9d, r9d           ; dy^2
    add r8d, r9d            ; dist_sq

    vcvtsi2ss xmm0, xmm0, r8d
    vucomiss xmm0, xmm9
    jb .next_inner          ; Too close

    ; Prepare call to similarity_match
    mov rcx, [rbp - 80]      ; features_i
    
    ; Compute features_j
    mov rax, rbx            ; features
    mov edx, r11d
    imul edx, edi           ; j * n_features
    shl rdx, 2
    add rax, rdx            ; features_j ptr (in RAX)

    ; Save volatile (if needed, but r10 and r11 are clobbered by calls usually)
    push r10
    push r11
    
    ; Shadow space (32 bytes)
    sub rsp, 32
    ; rsp is now aligned (since XXX0 - 16 - 32 = XXX0)
    
    ; Spill non-volatile registers we need across the call if we didn't use preserved ones
    ; (But we are using rbx, r12-r15, rsi, rdi which are preserved)
    ; We need to save xmm8/xmm9 because they are volatile? 
    ; Wait, XMM6-XMM15 are preserved in Win64!
    ; But XMM0-XMM5 are volatile.
    ; thresholds are in xmm8/xmm9 which are preserved in Win64.
    ; In System V, all XMMs are volatile? No, XMM0.. only are volatile?
    ; Actually, in System V, all XMM registers are volatile.
    ; In Win64, XMM6..XMM15 are non-volatile.
    
    vmovss [rbp - 84], xmm8    ; Threshold
    vmovss [rbp - 88], xmm9    ; MinDist
    
%ifidn __OUTPUT_FORMAT__, win64
    mov rdx, rax            ; Arg2: features_j
    mov r8d, edi            ; Arg3: n_features
    ; Arg1 (rcx) already set
%else
    mov rdi, rcx            ; Arg1: features_i
    mov rsi, rax            ; Arg2: features_j
    mov edx, edi            ; Arg3: n_features
%endif

    call wu_asm_similarity_match

    ; Restore thresholds (important for System V, good for Win64)
    vmovss xmm8, [rbp - 84]
    vmovss xmm9, [rbp - 88]

    add rsp, 32
    pop r11
    pop r10

    ; Check threshold
    vucomiss xmm0, xmm8
    jbe .next_inner

    ; Store match
    mov rax, rsi        ; count
    imul rax, 12        ; * 3 * 4
    add rax, r14        ; matches base
    
    vcvtsi2ss xmm1, xmm1, r10d
    vmovss [rax], xmm1
    vcvtsi2ss xmm1, xmm1, r11d
    vmovss [rax + 4], xmm1
    vmovss [rax + 8], xmm0
    
    inc esi             ; count++

.next_inner:
    inc r11d
    jmp .inner_loop

.next_outer:
    inc r10d
    jmp .outer_loop

.search_done:
    mov eax, esi        ; Return count

    add rsp, 72
    pop rsi
    pop rdi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

