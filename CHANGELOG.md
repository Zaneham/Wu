# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-20

### Added
- **Native SIMD Acceleration**: Initial release of x86-64 AVX2 optimized computational kernels for core forensic algorithms.
    - `copymove.asm`: Optimized block matching for copy-move detection (~20x speedup).
    - `prnu.asm`: Optimized PCE (Peak-to-Correlation Energy) calculation (~7x speedup).
    - `blockgrid.asm`: Optimized JPEG grid inconsistency detection (~8x speedup).
    - `lighting.asm`: Optimized light source direction estimation (~6.5x speedup).
- **Win64 ABI Compliance**: Full support for Windows 64-bit calling conventions, including shadow space handling and 16-byte stack alignment.
- **NumPy-Aware JSON Serialization**: Custom `WuJSONEncoder` in `state.py` to handle NumPy types during analysis export.

### Changed
- **CLI Robustness**: 
    - Forced UTF-8 encoding for stdout on Windows to prevent locale-specific JSON corruption.
    - Silenced `Pillow` deprecation warnings by migrating to `getexif()`.
    - Removed internal debug prints that interfered with structured tool output.
- **E2E Testing**: Standardized `PYTHONPATH` handling in CLI tests for better environment isolation.

### Fixed
- Fatal crash in `copymove` assembly due to incorrect stack pointer alignment on Windows.
- Buffer overflow risk in C-to-Assembly wrappers by enforcing strict size checks.
- JSON output corruption in CLI when running with parallelism enabled.

## [1.0.2] - 2025-12-19
### Fixed
- Minor bug fixes in metadata extraction.
- Improved error handling for corrupt JPEG headers.
