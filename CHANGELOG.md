# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.1] - 2025-12-23

### Fixed
- **AI Signature False Positives**: Fixed bug where AI generator signatures (e.g. "flux") were incorrectly detected in binary metadata blobs like Photoshop ImageSourceData. Now only scans text-based metadata fields with >90% printable characters.

### Changed
- **British English in Assembly**: Corrected US English spellings in assembly file comments (optimised, normalised, initialised).

## [1.5.0] - 2025-12-23

### Added
- **Authenticity Burden Mode**: New `--authenticity-mode` flag inverts the epistemic framing from "prove it's fake" to "prove it's authentic". Useful for supply chain verification and legal chain of custody.
  - New assessment states: `VERIFIED_AUTHENTIC`, `LIKELY_AUTHENTIC`, `UNVERIFIED`, `INSUFFICIENT_DATA`, `AUTHENTICITY_COMPROMISED`
  - Tracks verification chain (positive provenance evidence) and provenance gaps (missing verification)
  - Confidence scoring based on weighted verification dimensions (C2PA, PRNU, metadata, quantisation)
- **Authenticity Section in PDF Reports**: Court-ready reports now include authenticity assessment when running in authenticity mode.

## [1.4.0] - 2025-12-22

### Added
- **Lip-Sync Analysis**: Deterministic audio-visual synchronisation detection for deepfake identification.
  - Q15 fixed-point FFT for bit-exact reproducibility across platforms
  - Formant extraction using LPC analysis
  - Phoneme-to-viseme correlation mapping
  - Temporal offset detection with configurable thresholds

## [1.3.0] - 2025-12-21

### Added
- **Correlation Warnings**: Cross-dimensional analysis that identifies when findings from different forensic dimensions corroborate or contradict each other.
  - Warns when metadata claims authenticity but visual analysis shows manipulation
  - Highlights corroborating evidence across dimensions
- **Voice/Audio Analysis**: Enhanced audio forensics with spectral discontinuity detection and ENF (Electric Network Frequency) analysis.

### Fixed
- Native DLL now optional in PyInstaller builds for cross-platform compatibility.
- Video dimension properly included in `WuAnalysis.dimensions` and `to_dict()` output.

## [1.2.0] - 2025-12-20

### Added
- **Native Video Forensics**: Integrated H.264/MJPEG bitstream analysis for container anomalies and codec-level splicing markers.
- **Cross-Modal Analysis**: Correlates findings between audio and video streams to identify temporal inconsistencies.
- **Standalone CLI Executable**: Windows users can now download `wu.exe` without requiring Python installation.
- **GitHub Actions Release Workflow**: Automated PyPI publishing and executable builds.

### Changed
- Unified version management to `pyproject.toml`.
- Reduced false positive rates across analysers:
  - Metadata: Stripped metadata no longer flagged as suspicious (93% → 0% FPR)
  - Lighting: Conservative thresholds for court admissibility (48% → 4% FPR)
  - Copy-move: Stricter DCT similarity thresholds (36% → 7% FPR on CASIA)

## [1.1.0] - 2025-12-20

### Added
- **Native SIMD Acceleration**: Zane's initial release of x86-64 AVX2 optimised computational kernels for core forensic algorithms.
    - `copymove.asm`: optimised block matching for copy-move detection (~20x speedup).
    - `prnu.asm`: optimised PCE (Peak-to-Correlation Energy) calculation (~7x speedup).
    - `blockgrid.asm`:  optimised JPEG grid inconsistency detection (~8x speedup).
    - `lighting.asm`:  optimised light source direction estimation (~6.5x speedup).
- **Win64 ABI Compliance**:  full support for Windows 64-bit calling conventions, including shadow space handling and 16-byte stack alignment.
- **NumPy-Aware JSON Serialization**:  custom `WuJSONEncoder` in `state.py` to handle NumPy types during analysis export.

### Changed
- **CLI Robustness**: 
    - Forced UTF-8 encoding for stdout on Windows to prevent locale-specific JSON corruption.
    - Silenced `Pillow` deprecation warnings by migrating to `getexif()`.
    - Removed internal debug prints that interfered with structural tool output.
- **E2E Testing**:  standardised `PYTHONPATH` handling in CLI tests for better environment isolation.

### Fixed
- Fatal crash in `copymove` assembly due to incorrect stack pointer alignment on Windows, resolved by Zane.
- Buffer overflow risk in C-to-Assembly wrappers by enforcing strict size checks.
- JSON output corruption in CLI when running with parallelism enabled.

## [1.0.2] - 2025-12-19
### Fixed
- Minor bug fixes in metadata extraction.
- Improved error handling for corrupt JPEG headers.
