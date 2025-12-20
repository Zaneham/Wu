# 20/12/2025

# Project Dependencies and Vendored Logic

This document provides a comprehensive list of all external dependencies and vendored components utilised by the Wu Forensics toolkit. This manifest is maintained to ensure forensic reproducibility and chain-of-evidence integrity.


## Core Dependencies

These packages are required for the fundamental operation of the toolkit.

| Dependency       | Purpose          |  VersionRequirement |               
| :--- | :--- | :--- |
| `exifread` | Metadata extraction from image files. | `>=3.0` |
| `python-magic` | File type identification via libmagic. | `>=0.4` |
| `pillow` | Core image processing and manipulation. | `>=10.0` |
| `reportlab` | Tooling for PDF report generation. | `>=4.0` |
| `jinja2` | Templating engine for structured forensic reports. | `>=3.1` |
| `click` | Command-line interface orchestration. | `>=8.0` |


## Optional / Dimension-Specific Dependencies

These packages are utilised for specific forensic dimensions and may be installed via extras (e.g., `pip install wu-forensics[video]`).

### Video Forensic Dimension
| Dependency | Purpose | VersionRequirement |
| :--- | :--- | :--- |
| `opencv-python` | Foundation for video stream handling and basic CV. | `>=4.8` |
| `numpy` | Numerical array operations for frame analysis. | `>=1.24` |

### Audio Forensic Dimension (ENF)
| Dependency | Purpose | VersionRequirement |
| :--- | :--- | :--- |
| `librosa` | Audio feature extraction and STFT analysis. | `>=0.10` |
| `soundfile` | Native reading of audio bitstreams. | `>=0.12` |

### Machine Learning & Provenance
| Dependency | Purpose | VersionRequirement |
| :--- | :--- | :--- |
| `onnxruntime` | Inference engine for deep-learning forensic models. | `>=1.16` |
| `c2pa-python` | Verification of Content Credentials and provenance data. | `>=0.4` |


## Vendored Components

Components directly embedded into the source tree to ensure bit-exact reproducibility and avoid version drift in external decoders/libraries.

### [pocketfft](https://github.com/mreineck/pocketfft)

- **Location**: `src/wu/vendor/pocketfft/`
- **Purpose**: Fast Fourier Transform (FFT) for frequency-domain forensic algorithms (Block Grid, ENF).
- **Status**: VENDORED
- **SHA-256 Hash**: `4983075FFEFBEAF02E97C9AAAC36DF44FB59B977AC8BF38C9EFDEDD9BA3E63BD`
- **Technical Note**: Wu utilises the `pocketfft_hdronly.h` implementation (hayguen/pocketfft @ cpp branch) for all frequency-domain operations to ensure forensic consistency.


## Developer / Verification Tooling

| Dependency | Purpose | VersionRequirement |
| :--- | :--- | :--- |
| `pytest` | Test suite orchestration. | `>=7.0` |
| `pytest-cov` | Coverage analysis for verification reports. | `>=4.0` |
| `hatchling` | Build backend for secure distribution. | Current |
| `twine` | Secure PyPI distribution tool. | Current |
