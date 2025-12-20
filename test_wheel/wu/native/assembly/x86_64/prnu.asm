; Wu Forensics - PRNU Analysis Assembly
; x86-64 AVX2/FMA implementation

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

; Data section first (for forward references)
section .data align=32

wu_prnu_inf_double: dq 0x7FF0000000000000

section .text

extern wu_asm_hsum_f64

; ============================================================================
; CORRELATION SUM
; ============================================================================
global wu_asm_correlation_sum
wu_asm_correlation_sum:
    push rbp
    mov rbp, rsp
    push rbx
    mov r10, ARG1
    mov r11, ARG2
    mov rcx, ARG3
    vxorpd ymm0, ymm0, ymm0
    vxorpd ymm1, ymm1, ymm1
    cmp rcx, 8
    jl .corr_small
.corr_loop:
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
    jge .corr_loop
    vaddpd ymm0, ymm0, ymm1
.corr_small:
    cmp rcx, 4
    jl .corr_scalar
    vmovupd ymm2, [r10]
    vmovupd ymm3, [r11]
    vfmadd231pd ymm0, ymm2, ymm3
    add r10, 32
    add r11, 32
    sub rcx, 4

.corr_scalar:
    vextractf128 xmm2, ymm0, 1
    vaddpd xmm0, xmm0, xmm2
    vhaddpd xmm0, xmm0, xmm0
    test rcx, rcx
    jz .corr_done
.corr_remainder:
    vmovsd xmm2, [r10]
    vmovsd xmm3, [r11]
    vfmadd231sd xmm0, xmm2, xmm3
    add r10, 8
    add r11, 8
    dec rcx
    jnz .corr_remainder
.corr_done:
    vzeroupper
    pop rbx
    pop rbp
    ret

; ============================================================================
; MEAN VARIANCE
; ============================================================================
global wu_asm_mean_variance
wu_asm_mean_variance:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    mov r10, ARG1
    mov rbx, r10     ; Backup data ptr to RBX
    mov r11, ARG2
    mov r12, ARG3
    mov r13, ARG4
    vxorpd ymm0, ymm0, ymm0
    mov rcx, r11
    cmp rcx, 4
    jl .mv_scalar
.mv_sum_loop:
    vmovupd ymm1, [r10]
    vaddpd ymm0, ymm0, ymm1
    add r10, 32
    sub rcx, 4
    cmp rcx, 4
    jge .mv_sum_loop
.mv_scalar:
    vextractf128 xmm2, ymm0, 1
    vaddpd xmm0, xmm0, xmm2
    vhaddpd xmm0, xmm0, xmm0
    test rcx, rcx
    jz .mv_compute_mean
    jz .mv_compute_mean

.mv_remainder:
    vmovsd xmm1, [r10]
    vaddsd xmm0, xmm0, xmm1
    add r10, 8
    dec rcx
    jnz .mv_remainder
.mv_compute_mean:
    vcvtsi2sd xmm1, xmm1, r11
    vdivsd xmm0, xmm0, xmm1
    vmovsd [r12], xmm0
    vmovsd [r12], xmm0
    mov r10, rbx     ; Restore data ptr from RBX
    mov rcx, r11
    vxorpd ymm2, ymm2, ymm2
    vbroadcastsd ymm3, xmm0
    cmp rcx, 4
    jl .mv_var_scalar
.mv_var_loop:
    vmovupd ymm1, [r10]
    vsubpd ymm4, ymm1, ymm3
    vfmadd231pd ymm2, ymm4, ymm4
    add r10, 32
    sub rcx, 4
    cmp rcx, 4
    jge .mv_var_loop
.mv_var_scalar:
    vextractf128 xmm4, ymm2, 1
    vaddpd xmm2, xmm2, xmm4
    vhaddpd xmm2, xmm2, xmm2

    test rcx, rcx
    jz .mv_finalize

.mv_var_remainder:
    vmovsd xmm4, [r10]
    vsubsd xmm4, xmm4, xmm0
    vfmadd231sd xmm2, xmm4, xmm4
    add r10, 8
    dec rcx
    jnz .mv_var_remainder
.mv_finalize:
    vcvtsi2sd xmm1, xmm1, r11
    vdivsd xmm2, xmm2, xmm1
    vmovsd [r13], xmm2
    vzeroupper
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; FIND PEAK
; ============================================================================
global wu_asm_find_peak
wu_asm_find_peak:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    mov r10, ARG1
    mov r11, ARG2
    mov r12, ARG3
    vmovsd xmm0, [r10]
    xor eax, eax
    mov rcx, 1
.peak_loop:
    cmp rcx, r11
    jge .peak_done
    mov rbx, rcx
    shl rbx, 3
    vmovsd xmm1, [r10 + rbx]
    vucomisd xmm1, xmm0
    jbe .peak_next
    vmovsd xmm0, xmm1
    mov rax, rcx
.peak_next:
    inc rcx
    jmp .peak_loop
.peak_done:
    mov [r12], rax
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; COMPUTE PCE
; ============================================================================
global wu_asm_compute_pce
wu_asm_compute_pce:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov [rbp - 8], ARG1
    mov [rbp - 16], ARG2
    mov [rbp - 24], ARG3
    mov [rbp - 32], ARG4
%ifidn __OUTPUT_FORMAT__, win64
    mov eax, [rbp + 0x30]
    mov [rbp - 36], eax
    mov eax, [rbp + 0x38]
    mov [rbp - 40], eax
%else
    mov eax, [rbp + 0x10]
    mov [rbp - 36], eax
    mov eax, [rbp + 0x18]
    mov [rbp - 40], eax
%endif
    mov rax, [rbp - 36]
    imul rax, [rbp - 16]
    add rax, [rbp - 32]
    shl rax, 3
    mov rcx, [rbp - 8]
    vmovsd xmm14, [rcx + rax]
    vmulsd xmm15, xmm14, xmm14
    vxorpd xmm0, xmm0, xmm0
    xor r12d, r12d
    xor r10d, r10d
.pce_y_loop:
    cmp r10d, [rbp - 24]
    jge .pce_compute
    xor r11d, r11d
.pce_x_loop:
    cmp r11d, [rbp - 16]
    jge .pce_next_y
    mov eax, r11d
    sub eax, [rbp - 32]
    cdq
    xor eax, edx
    sub eax, edx
    cmp eax, [rbp - 40]
    jge .pce_include
    mov eax, r10d
    sub eax, [rbp - 36]
    cdq
    xor eax, edx
    sub eax, edx
    cmp eax, [rbp - 40]
    jl .pce_next_x
.pce_include:
    mov rax, r10
    imul rax, [rbp - 16]
    add rax, r11
    shl rax, 3
    mov rcx, [rbp - 8]
    vmovsd xmm1, [rcx + rax]
    vfmadd231sd xmm0, xmm1, xmm1
    inc r12d
.pce_next_x:
    inc r11d
    jmp .pce_x_loop
.pce_next_y:
    inc r10d
    jmp .pce_y_loop
.pce_compute:
    test r12d, r12d
    jz .pce_zero
    vcvtsi2sd xmm1, xmm1, r12d
    vdivsd xmm0, xmm0, xmm1
    vdivsd xmm0, xmm15, xmm0
    jmp .pce_done
.pce_zero:
    vmovsd xmm0, [rel wu_prnu_inf_double]
.pce_done:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
