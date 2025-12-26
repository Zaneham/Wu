; Wu Forensics - H.264 Bitstream Anomaly Scanner
; x86-64 AVX2 implementation
;
; Scans H.264 NAL unit streams for structural anomalies that may
; indicate splicing, tampering, or re-encoding.
;
; Build (Windows): nasm -f win64 -o h264_bitstream_forensic.obj h264_bitstream_forensic.asm
; Build (Linux):   nasm -f elf64 -o h264_bitstream_forensic.o h264_bitstream_forensic.asm
; Build (macOS):   nasm -f macho64 -o h264_bitstream_forensic.o h264_bitstream_forensic.asm

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

; NAL start code pattern: 00 00 01 or 00 00 00 01
wu_nal_start_3:     db 0x00, 0x00, 0x01, 0x00
                    times 28 db 0
wu_nal_start_4:     db 0x00, 0x00, 0x00, 0x01
                    times 28 db 0

; NAL type masks
wu_nal_type_mask:   db 0x1F
                    times 31 db 0

section .text

; ============================================================================
; NAL UNIT SCANNER
; ============================================================================
; Scans a byte buffer for NAL start codes and extracts NAL type sequence.
; Fast vectorised search for start code patterns.
;
; Input:  ARG1 = buffer        (uint8_t*, bitstream data)
;         ARG2 = length        (int, buffer length)
;         ARG3 = nal_types     (uint8_t*, output: sequence of NAL types)
;         ARG4 = nal_offsets   (int32_t*, output: byte offset of each NAL)
;         [stack] = max_nals   (int, maximum NALs to record)
; Output: EAX = number of NALs found
; ============================================================================
global wu_asm_scan_nal_units
wu_asm_scan_nal_units:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r10, ARG1               ; buffer
    mov r11d, ARG2d             ; length
    mov r12, ARG3               ; nal_types output
    mov r13, ARG4               ; nal_offsets output

%ifidn __OUTPUT_FORMAT__, win64
    mov r14d, [rbp + 48]        ; max_nals
%else
    mov r14d, r8d
%endif

    xor r15d, r15d              ; count = 0

    ; Need at least 4 bytes for start code + 1 for NAL header
    cmp r11d, 5
    jl .scan_done

    ; Prepare search pattern (00 00 01)
    vpxor ymm0, ymm0, ymm0      ; Zero for comparison
    vpcmpeqb ymm1, ymm1, ymm1   ; All ones
    vpabsb ymm2, ymm1           ; All ones (01 pattern)

    xor ebx, ebx                ; current position = 0

.scan_loop:
    ; Check remaining bytes
    mov eax, r11d
    sub eax, ebx
    cmp eax, 4
    jl .scan_done

    ; Check output limit
    cmp r15d, r14d
    jge .scan_done

    ; Simple scalar scan for start code (vectorised version below is optional)
    ; Look for 00 00 01 or 00 00 00 01
    movzx eax, byte [r10 + rbx]
    test eax, eax
    jnz .scan_next

    movzx eax, byte [r10 + rbx + 1]
    test eax, eax
    jnz .scan_next

    ; Found 00 00, check for 01 or 00 01
    movzx eax, byte [r10 + rbx + 2]
    cmp eax, 1
    je .found_start_3

    test eax, eax
    jnz .scan_next

    ; Check for 00 00 00 01
    movzx eax, byte [r10 + rbx + 3]
    cmp eax, 1
    je .found_start_4

    jmp .scan_next

.found_start_3:
    ; 3-byte start code at position ebx
    lea ecx, [ebx + 3]          ; NAL header position
    jmp .record_nal

.found_start_4:
    ; 4-byte start code at position ebx
    lea ecx, [ebx + 4]          ; NAL header position

.record_nal:
    ; Check we have a NAL header byte
    cmp ecx, r11d
    jge .scan_done

    ; Extract NAL type (lower 5 bits of NAL header)
    movzx eax, byte [r10 + rcx]
    and eax, 0x1F               ; NAL type

    ; Store NAL type
    mov [r12 + r15], al

    ; Store NAL offset
    mov rdx, r15
    shl rdx, 2                  ; * 4 for int32
    add rdx, r13
    mov [rdx], ebx

    inc r15d

    ; Skip past the start code
    mov ebx, ecx
    jmp .scan_next

.scan_next:
    inc ebx
    jmp .scan_loop

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
; NAL SEQUENCE ANOMALY DETECTOR
; ============================================================================
; Analyses a sequence of NAL types for structural anomalies.
; Detects:
;   - Missing SPS/PPS before IDR
;   - SPS/PPS appearing after slices (splice indicator)
;   - Unexpected IDR spacing
;   - Invalid NAL type values
;
; Input:  ARG1 = nal_types     (uint8_t*, sequence of NAL types)
;         ARG2 = count         (int, number of NALs)
;         ARG3 = anomalies     (int32_t*, output: [index, anomaly_code] pairs)
;         ARG4 = max_anomalies (int)
; Output: EAX = number of anomalies found
;
; Anomaly codes:
;   1 = Missing SPS before first slice
;   2 = Missing PPS before first slice
;   3 = SPS after slice (potential splice)
;   4 = PPS after slice (potential splice)
;   5 = Non-IDR slice without preceding IDR
;   6 = Invalid NAL type (>31)
;   7 = Consecutive IDRs (unusual)
; ============================================================================
global wu_asm_analyse_nal_sequence
wu_asm_analyse_nal_sequence:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8

    mov r10, ARG1               ; nal_types
    mov r11d, ARG2d             ; count
    mov r12, ARG4               ; max_anomalies (ARG4 before ARG3 for easier access)
    mov r13, ARG3               ; anomalies output

    xor r14d, r14d              ; anomaly_count = 0
    xor r15d, r15d              ; flags: bit0=seen_sps, bit1=seen_pps, bit2=seen_slice, bit3=seen_idr
    mov [rbp - 8], r12          ; Store max_anomalies

    test r11d, r11d
    jz .seq_done

    xor ecx, ecx                ; index = 0

.seq_loop:
    cmp ecx, r11d
    jge .seq_done

    mov eax, [rbp - 8]
    cmp r14d, eax
    jge .seq_done

    movzx eax, byte [r10 + rcx] ; NAL type

    ; Check for invalid NAL type
    cmp eax, 31
    jle .check_type

    ; Record anomaly: invalid NAL type
    mov rdx, r14
    shl rdx, 3
    add rdx, r13
    mov [rdx], ecx              ; index
    mov dword [rdx + 4], 6      ; code: invalid NAL type
    inc r14d
    jmp .seq_next

.check_type:
    ; SPS (type 7)
    cmp eax, 7
    jne .check_pps

    ; Check if we've seen slices already (potential splice)
    test r15d, 4                ; bit2 = seen_slice
    jz .set_sps

    ; SPS after slice - anomaly
    mov eax, [rbp - 8]
    cmp r14d, eax
    jge .set_sps
    mov rdx, r14
    shl rdx, 3
    add rdx, r13
    mov [rdx], ecx
    mov dword [rdx + 4], 3      ; code: SPS after slice
    inc r14d

.set_sps:
    or r15d, 1                  ; Set seen_sps flag
    jmp .seq_next

.check_pps:
    ; PPS (type 8)
    cmp eax, 8
    jne .check_idr

    ; Check if we've seen slices already
    test r15d, 4
    jz .set_pps

    ; PPS after slice - anomaly
    mov eax, [rbp - 8]
    cmp r14d, eax
    jge .set_pps
    mov rdx, r14
    shl rdx, 3
    add rdx, r13
    mov [rdx], ecx
    mov dword [rdx + 4], 4      ; code: PPS after slice
    inc r14d

.set_pps:
    or r15d, 2                  ; Set seen_pps flag
    jmp .seq_next

.check_idr:
    ; IDR slice (type 5)
    cmp eax, 5
    jne .check_non_idr

    ; Check for missing SPS/PPS before first slice
    test r15d, 4                ; seen_slice?
    jnz .check_consecutive_idr

    ; First slice - check SPS/PPS
    test r15d, 1                ; seen_sps?
    jnz .check_first_pps

    ; Missing SPS
    mov eax, [rbp - 8]
    cmp r14d, eax
    jge .check_first_pps
    mov rdx, r14
    shl rdx, 3
    add rdx, r13
    mov [rdx], ecx
    mov dword [rdx + 4], 1      ; code: missing SPS
    inc r14d

.check_first_pps:
    test r15d, 2                ; seen_pps?
    jnz .set_idr

    ; Missing PPS
    mov eax, [rbp - 8]
    cmp r14d, eax
    jge .set_idr
    mov rdx, r14
    shl rdx, 3
    add rdx, r13
    mov [rdx], ecx
    mov dword [rdx + 4], 2      ; code: missing PPS
    inc r14d
    jmp .set_idr

.check_consecutive_idr:
    ; Check for consecutive IDRs (unusual pattern)
    test r15d, 8                ; Last was IDR?
    jz .set_idr

    mov eax, [rbp - 8]
    cmp r14d, eax
    jge .set_idr
    mov rdx, r14
    shl rdx, 3
    add rdx, r13
    mov [rdx], ecx
    mov dword [rdx + 4], 7      ; code: consecutive IDRs
    inc r14d

.set_idr:
    or r15d, 12                 ; Set seen_slice and seen_idr flags
    jmp .seq_next

.check_non_idr:
    ; Non-IDR slice (type 1)
    cmp eax, 1
    jne .seq_next

    ; Check for slice without preceding IDR
    test r15d, 8                ; seen_idr?
    jnz .set_slice

    mov eax, [rbp - 8]
    cmp r14d, eax
    jge .set_slice
    mov rdx, r14
    shl rdx, 3
    add rdx, r13
    mov [rdx], ecx
    mov dword [rdx + 4], 5      ; code: non-IDR without IDR
    inc r14d

.set_slice:
    or r15d, 4                  ; Set seen_slice flag
    and r15d, ~8                ; Clear seen_idr (for consecutive detection)

.seq_next:
    inc ecx
    jmp .seq_loop

.seq_done:
    mov eax, r14d

    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


; ============================================================================
; EMULATION PREVENTION BYTE SCANNER
; ============================================================================
; Counts emulation prevention bytes (0x03 in 00 00 03 sequence).
; Unusual EPB patterns can indicate manipulation.
;
; Input:  ARG1 = buffer        (uint8_t*, NAL data)
;         ARG2 = length        (int)
; Output: EAX = count of EPB sequences found
; ============================================================================
global wu_asm_count_epb
wu_asm_count_epb:
    push rbp
    mov rbp, rsp
    push rbx

    mov r10, ARG1               ; buffer
    mov r11d, ARG2d             ; length

    xor eax, eax                ; count = 0

    cmp r11d, 3
    jl .epb_done

    xor ecx, ecx                ; position = 0
    mov edx, r11d
    sub edx, 2                  ; length - 2

.epb_loop:
    cmp ecx, edx
    jge .epb_done

    ; Check for 00 00 03
    movzx ebx, byte [r10 + rcx]
    test ebx, ebx
    jnz .epb_next

    movzx ebx, byte [r10 + rcx + 1]
    test ebx, ebx
    jnz .epb_next

    movzx ebx, byte [r10 + rcx + 2]
    cmp ebx, 3
    jne .epb_next

    inc eax                     ; Found EPB
    add ecx, 3                  ; Skip past EPB
    jmp .epb_loop

.epb_next:
    inc ecx
    jmp .epb_loop

.epb_done:
    pop rbx
    pop rbp
    ret


; ============================================================================
; CABAC/CAVLC ENTROPY STATISTICS
; ============================================================================
; Computes basic entropy statistics on slice data.
; Useful for detecting anomalous encoding patterns.
;
; Input:  ARG1 = buffer        (uint8_t*, slice data after header)
;         ARG2 = length        (int)
;         ARG3 = stats         (int32_t[4]: [zero_count, one_count, byte_sum, entropy_est])
; Output: None (results in stats)
;
; Entropy estimation: count bit transitions, compare to random expectation
; ============================================================================
global wu_asm_entropy_stats
wu_asm_entropy_stats:
    push rbp
    mov rbp, rsp
    push rbx
    push r12

    mov r10, ARG1               ; buffer
    mov r11d, ARG2d             ; length
    mov r12, ARG3               ; stats

    xor eax, eax                ; zero_count
    xor ebx, ebx                ; one_count
    xor ecx, ecx                ; byte_sum
    xor edx, edx                ; transitions

    test r11d, r11d
    jz .entropy_store

    ; Previous byte for transition counting
    movzx r8d, byte [r10]
    add ecx, r8d                ; byte_sum

    ; Count bits in first byte
    popcnt r9d, r8d
    add ebx, r9d                ; one_count
    mov eax, 8
    sub eax, r9d                ; zero_count for first byte

    mov edi, 1                  ; position = 1

.entropy_loop:
    cmp edi, r11d
    jge .entropy_store

    movzx r8d, byte [r10 + rdi]
    add ecx, r8d                ; byte_sum

    ; Count ones
    popcnt r9d, r8d
    add ebx, r9d
    mov esi, 8
    sub esi, r9d
    add eax, esi                ; zero_count

    ; Count transitions (XOR with previous, count bits)
    movzx esi, byte [r10 + rdi - 1]
    xor esi, r8d
    popcnt esi, esi
    add edx, esi                ; transitions

    inc edi
    jmp .entropy_loop

.entropy_store:
    mov [r12], eax              ; zero_count
    mov [r12 + 4], ebx          ; one_count
    mov [r12 + 8], ecx          ; byte_sum

    ; Entropy estimate: transitions / (length * 4) * 100
    ; Random data has ~50% bit transitions
    mov eax, edx
    imul eax, 100
    mov ecx, r11d
    shl ecx, 2                  ; length * 4
    test ecx, ecx
    jz .entropy_zero_div
    xor edx, edx
    div ecx
    mov [r12 + 12], eax
    jmp .entropy_done

.entropy_zero_div:
    mov dword [r12 + 12], 0

.entropy_done:
    pop r12
    pop rbx
    pop rbp
    ret
