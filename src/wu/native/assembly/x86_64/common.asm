; Wu Forensics - Common Assembly Utilities (Header Only)
; x86-64 AVX2/FMA implementation
;
; This file contains shared utility MACROS and DEFINES used by other modules.
; Target: x86-64 with AVX2 and FMA support

; ============================================================================
; MACROS
; ============================================================================

; PUSH_XMM n - Saves n non-volatile XMM registers (starting from XMM6)
; used in Windows x64 calling convention.
%macro PUSH_XMM 1
    sub rsp, %1 * 16
    %assign %%i 0
    %rep %1
        %assign %%reg %%i + 6
        vmovups [rsp + %%i * 16], xmm%+ %%reg
        %assign %%i %%i + 1
    %endrep
%endmacro

; POP_XMM n - Restores n non-volatile XMM registers
%macro POP_XMM 1
    %assign %%i 0
    %rep %1
        %assign %%reg %%i + 6
        vmovups xmm%+ %%reg, [rsp + %%i * 16]
        %assign %%i %%i + 1
    %endrep
    add rsp, %1 * 16
%endmacro

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

SECTION_DATA
align 32
const_one_f32: times 8 dd 1.0
const_zero_f32: times 8 dd 0.0
const_half_f32: times 8 dd 0.5
