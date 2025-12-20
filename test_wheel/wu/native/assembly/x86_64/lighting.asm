; Wu Forensics - Lighting Analysis Assembly
; x86-64 AVX2/FMA implementation
;
; Light direction estimation from image gradients.
; Uses shape-from-shading principles.

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

wu_light_scale_factor: dq 0.5
wu_light_one: dq 1.0

section .text

; ============================================================================
; WEIGHTED GRADIENT STATS
; ============================================================================
global wu_asm_weighted_gradient_stats
wu_asm_weighted_gradient_stats:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    
    mov r10, ARG1
    mov r11, ARG2
    mov r12, ARG3
    mov r13, ARG4
    
%ifidn __OUTPUT_FORMAT__, win64
    mov r14, [rbp + 0x30]
    mov rbx, [rbp + 0x38]
%else
    mov r14, [rbp + 0x10]
    mov rbx, [rbp + 0x18]
%endif
    
    vxorpd ymm0, ymm0, ymm0
    vxorpd ymm1, ymm1, ymm1
    vxorpd ymm2, ymm2, ymm2
    
    mov rcx, r12
    cmp rcx, 4
    jl .grad_scalar

.grad_loop:
    vmovupd ymm3, [r10]
    vmovupd ymm4, [r11]
    vmulpd ymm5, ymm3, ymm3
    vfmadd231pd ymm5, ymm4, ymm4
    vsqrtpd ymm6, ymm5
    vfmadd231pd ymm0, ymm3, ymm6
    vfmadd231pd ymm1, ymm4, ymm6
    vaddpd ymm2, ymm2, ymm6
    add r10, 32
    add r11, 32
    sub rcx, 4
    cmp rcx, 4
    jge .grad_loop

.grad_scalar:
    vperm2f128 ymm5, ymm0, ymm0, 0x01
    vaddpd ymm0, ymm0, ymm5
    vextractf128 xmm5, ymm0, 1
    vaddpd xmm0, xmm0, xmm5
    vhaddpd xmm0, xmm0, xmm0
    
    vperm2f128 ymm5, ymm1, ymm1, 0x01
    vaddpd ymm1, ymm1, ymm5
    vextractf128 xmm5, ymm1, 1
    vaddpd xmm1, xmm1, xmm5
    vhaddpd xmm1, xmm1, xmm1
    
    vperm2f128 ymm5, ymm2, ymm2, 0x01
    vaddpd ymm2, ymm2, ymm5
    vextractf128 xmm5, ymm2, 1
    vaddpd xmm2, xmm2, xmm5
    vhaddpd xmm2, xmm2, xmm2
    
    test rcx, rcx
    jz .grad_store

.grad_remainder:
    vmovsd xmm3, [r10]
    vmovsd xmm4, [r11]
    vmulsd xmm5, xmm3, xmm3
    vfmadd231sd xmm5, xmm4, xmm4
    vsqrtsd xmm6, xmm6, xmm5
    vfmadd231sd xmm0, xmm3, xmm6
    vfmadd231sd xmm1, xmm4, xmm6
    vaddsd xmm2, xmm2, xmm6
    add r10, 8
    add r11, 8
    dec rcx
    jnz .grad_remainder

.grad_store:
    vmovsd [r13], xmm0
    vmovsd [r14], xmm1
    vmovsd [rbx], xmm2
    vzeroupper
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; ESTIMATE LIGHT DIRECTION
; ============================================================================
global wu_asm_estimate_light_direction
wu_asm_estimate_light_direction:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    
    mov [rbp - 8], ARG4

%ifidn __OUTPUT_FORMAT__, win64
    mov rax, [rbp + 0x30]
    mov [rbp - 16], rax
    mov rax, [rbp + 0x38]
    mov [rbp - 24], rax
%else
    mov rax, [rbp + 0x10]
    mov [rbp - 16], rax
    mov rax, [rbp + 0x18]
    mov [rbp - 24], rax
%endif
    
    lea r12, [rbp - 32]
    lea r13, [rbp - 40]
    lea r14, [rbp - 48]
    
    mov ARG4, r12
%ifidn __OUTPUT_FORMAT__, win64
    mov [rsp + 0x20], r13
    mov [rsp + 0x28], r14
%else
    mov r8, r13
    mov r9, r14
%endif
    
    call wu_asm_weighted_gradient_stats
    
    vmovsd xmm0, [rbp - 32]
    vmovsd xmm1, [rbp - 40]
    vmovsd xmm2, [rbp - 48]
    
    vxorpd xmm3, xmm3, xmm3
    vucomisd xmm2, xmm3
    je .light_zero
    
    vdivsd xmm0, xmm0, xmm2
    vdivsd xmm1, xmm1, xmm2
    
    mov rax, [rbp - 8]
    vmovsd [rax], xmm0
    mov rax, [rbp - 16]
    vmovsd [rax], xmm1
    
    vmulsd xmm3, xmm0, xmm0
    vfmadd231sd xmm3, xmm1, xmm1
    vsqrtsd xmm3, xmm3, xmm3
    
    vmovsd xmm4, [rel wu_light_scale_factor]
    vdivsd xmm3, xmm3, xmm4
    
    vxorpd xmm4, xmm4, xmm4
    vmaxsd xmm3, xmm3, xmm4
    vmovsd xmm4, [rel wu_light_one]
    vminsd xmm3, xmm3, xmm4
    
    mov rax, [rbp - 24]
    vmovsd [rax], xmm3
    jmp .light_done

.light_zero:
    vxorpd xmm0, xmm0, xmm0
    mov rax, [rbp - 8]
    vmovsd [rax], xmm0
    mov rax, [rbp - 16]
    vmovsd [rax], xmm0
    mov rax, [rbp - 24]
    vmovsd [rax], xmm0

.light_done:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ============================================================================
; GRADIENT MAGNITUDE
; ============================================================================
global wu_asm_gradient_magnitude
wu_asm_gradient_magnitude:
    push rbp
    mov rbp, rsp
    
    mov r10, ARG1
    mov r11, ARG2
    mov rcx, ARG3
    mov r8, ARG4
    
    cmp rcx, 4
    jl .mag_scalar

.mag_loop:
    vmovupd ymm0, [r10]
    vmovupd ymm1, [r11]
    vmulpd ymm2, ymm0, ymm0
    vfmadd231pd ymm2, ymm1, ymm1
    vsqrtpd ymm3, ymm2
    vmovupd [r8], ymm3
    add r10, 32
    add r11, 32
    add r8, 32
    sub rcx, 4
    cmp rcx, 4
    jge .mag_loop

.mag_scalar:
    test rcx, rcx
    jz .mag_done

.mag_remainder:
    vmovsd xmm0, [r10]
    vmovsd xmm1, [r11]
    vmulsd xmm2, xmm0, xmm0
    vfmadd231sd xmm2, xmm1, xmm1
    vsqrtsd xmm3, xmm3, xmm2
    vmovsd [r8], xmm3
    add r10, 8
    add r11, 8
    add r8, 8
    dec rcx
    jnz .mag_remainder

.mag_done:
    vzeroupper
    pop rbp
    ret
