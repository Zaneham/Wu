class BitstreamReader:
    """
    Utility to read a bitstream bit-by-bit.
    Essential for parsing H.264 headers (Exp-Golomb codes).
    """
    def __init__(self, data: bytes):
        self.data = data
        self.byte_offset = 0
        self.bit_offset = 0 # 0 to 7 (MSB to LSB)

    def read_bit(self) -> int:
        """Read a single bit."""
        if self.byte_offset >= len(self.data):
            return 0
        
        bit = (self.data[self.byte_offset] >> (7 - self.bit_offset)) & 0x01
        self.bit_offset += 1
        if self.bit_offset == 8:
            self.bit_offset = 0
            self.byte_offset += 1
        return bit

    def read_bits(self, n: int) -> int:
        """Read n bits and return as integer."""
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def read_ue(self) -> int:
        """
        Read Unsigned Exponential-Golomb (ue(v)) code.
        Used extensively in H.264 headers.
        """
        leading_zeros = 0
        while self.read_bit() == 0 and leading_zeros < 32:
            leading_zeros += 1
        
        if leading_zeros == 0:
            return 0
        
        rem = self.read_bits(leading_zeros)
        return (1 << leading_zeros) - 1 + rem

    def read_se(self) -> int:
        """
        Read Signed Exponential-Golomb (se(v)) code.
        """
        code_num = self.read_ue()
        if code_num == 0:
            return 0
        
        # Map back to signed: 1->1, 2->-1, 3->2, 4->-2 ...
        sign = 1 if (code_num % 2 != 0) else -1
        return sign * ((code_num + 1) // 2)

    def byte_align(self):
        """Skip to next byte boundary."""
        if self.bit_offset != 0:
            self.bit_offset = 0
            self.byte_offset += 1

    @property
    def bytes_left(self) -> int:
        return len(self.data) - self.byte_offset
