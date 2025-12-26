# Wu API Reference

## Purpose

This document provides a comprehensive reference for the Wu forensic analysis toolkit's programming interface. It is intended for developers integrating Wu into automated pipelines, building custom forensic workflows, or extending the toolkit with additional analytical dimensions. The API is designed to surface structured uncertainty rather than binary classifications, enabling downstream systems to make informed decisions about the reliability of media evidence.

## Installation

Wu may be installed via pip from the Python Package Index:

```bash
# Basic installation
pip install wu-forensics

# With all optional dependencies
pip install "wu-forensics[all]"

# Specific feature sets
pip install "wu-forensics[video,audio]"
pip install "wu-forensics[c2pa]"
```

For development installations from source:

```bash
git clone https://github.com/Zaneham/wu.git
cd wu
pip install -e ".[all]"
```

## Core Classes

### WuAnalyzer

The primary entry point for forensic analysis. This class orchestrates multiple dimension analysers and aggregates their results into a unified report.

```python
from wu import WuAnalyzer

analyzer = WuAnalyzer(
    enable_metadata=True,      # EXIF and file metadata analysis
    enable_c2pa=True,          # Content credential verification
    enable_visual=True,        # Error Level Analysis (ELA)
    enable_enf=False,          # Electric Network Frequency analysis
    enable_copymove=False,     # Copy-move (clone) detection
    enable_prnu=False,         # Photo Response Non-Uniformity
    enable_blockgrid=False,    # JPEG block grid alignment
    enable_lighting=False,     # Lighting direction consistency
    enable_audio=False,        # Audio spectral forensics
    enable_thumbnail=False,    # EXIF thumbnail comparison
    enable_shadows=False,      # Shadow direction analysis
    enable_perspective=False,  # Vanishing point consistency
    enable_quantization=False, # JPEG quantisation table analysis
    enable_aigen=False,        # AI generation indicator detection
    enable_video=True,         # Native video bitstream analysis
    enable_lipsync=False,      # Audio-visual synchronisation
    parallel=True,             # Execute dimensions concurrently
    max_workers=None,          # Worker thread count (None = auto)
    authenticity_mode=False,   # Invert epistemic burden
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_metadata` | bool | True | Analyse EXIF headers, timestamps, GPS coordinates, and device attribution. |
| `enable_c2pa` | bool | True | Verify Content Authenticity Initiative (C2PA) credentials if present. |
| `enable_visual` | bool | True | Perform Error Level Analysis to detect compression inconsistencies. |
| `enable_enf` | bool | False | Detect Electric Network Frequency signatures in audio tracks. Requires audio content. |
| `enable_copymove` | bool | False | Detect duplicated regions within an image. Computationally intensive. |
| `enable_prnu` | bool | False | Analyse sensor fingerprint patterns. Computationally intensive. |
| `enable_blockgrid` | bool | False | Examine JPEG block grid alignment for splicing indicators. JPEG-specific. |
| `enable_lighting` | bool | False | Evaluate lighting direction consistency across image regions. |
| `enable_audio` | bool | False | Perform spectral discontinuity and noise floor analysis on audio. |
| `enable_thumbnail` | bool | False | Compare embedded EXIF thumbnail against main image content. |
| `enable_shadows` | bool | False | Analyse shadow directions for physical plausibility. |
| `enable_perspective` | bool | False | Examine vanishing point consistency. |
| `enable_quantization` | bool | False | Analyse JPEG quantisation tables for compression history. JPEG-specific. |
| `enable_aigen` | bool | False | Detect indicators of AI-generated content. |
| `enable_video` | bool | True | Analyse video container structure and codec-level markers. |
| `enable_lipsync` | bool | False | Detect audio-visual desynchronisation in video content. |
| `parallel` | bool | True | Execute dimension analysers concurrently using thread pool. |
| `max_workers` | int | None | Maximum worker threads. None selects automatically based on CPU count. |
| `authenticity_mode` | bool | False | Invert epistemic framing from "detect manipulation" to "verify authenticity". |

#### Methods

##### analyze(file_path: str) -> WuAnalysis

Perform forensic analysis on a single media file.

```python
result = analyzer.analyze("evidence.jpg")
print(result.overall)  # OverallAssessment enum
print(result.to_json())  # JSON string for serialisation
```

The method returns a `WuAnalysis` object containing per-dimension results, an aggregated overall assessment, and a human-readable findings summary. A SHA-256 hash of the analysed file is computed automatically for chain of custody purposes.

##### analyze_batch(file_paths: List[str], parallel_files: bool = True, max_file_workers: int = None) -> List[WuAnalysis]

Analyse multiple files with optional file-level parallelism.

```python
results = analyzer.analyze_batch([
    "photo1.jpg",
    "photo2.jpg",
    "video.mp4"
], parallel_files=True)

for result in results:
    print(f"{result.file_path}: {result.overall.value}")
```

Results are returned in the same order as the input file paths. Failed analyses produce `WuAnalysis` objects with `INSUFFICIENT_DATA` assessment rather than raising exceptions.

##### is_supported(file_path: str) -> bool

Check whether a file format is supported by Wu.

```python
if analyzer.is_supported("document.pdf"):
    result = analyzer.analyze("document.pdf")
```

##### get_supported_formats() -> List[str]

Return a list of supported file extensions.

```python
formats = WuAnalyzer.get_supported_formats()
# ['.jpg', '.jpeg', '.png', '.tiff', '.mp4', '.mov', ...]
```

---

### WuAnalysis

The complete result of analysing a media file. This dataclass contains per-dimension results, aggregated assessment, and supporting metadata for forensic reporting.

```python
from wu.state import WuAnalysis
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `file_path` | str | Absolute path to the analysed file. |
| `file_hash` | str | SHA-256 hash of the file for chain of custody verification. |
| `analyzed_at` | datetime | Timestamp when analysis was performed. |
| `wu_version` | str | Version of Wu used for the analysis. |
| `overall` | OverallAssessment | Aggregated assessment across all dimensions. |
| `findings_summary` | List[str] | Human-readable list of key findings. |
| `corroboration_summary` | str | Narrative describing convergent evidence across dimensions. |
| `correlation_warnings` | List[CorrelationWarning] | Cross-dimension conflicts detected. |
| `authenticity` | AuthenticityResult | Present only when `authenticity_mode=True`. |

##### Dimension Result Attributes

Each analysed dimension produces a `DimensionResult` accessible as an attribute:

| Attribute | Dimension |
|-----------|-----------|
| `metadata` | EXIF and file metadata |
| `visual` | Error Level Analysis |
| `c2pa` | Content credentials |
| `enf` | Electric Network Frequency |
| `copymove` | Clone detection |
| `prnu` | Sensor fingerprint |
| `blockgrid` | JPEG block alignment |
| `lighting` | Light direction |
| `audio` | Audio forensics |
| `thumbnail` | Thumbnail comparison |
| `shadows` | Shadow direction |
| `perspective` | Vanishing points |
| `quantization` | JPEG quantisation |
| `aigen` | AI generation indicators |
| `video` | Video bitstream analysis |
| `lipsync` | Audio-visual sync |

#### Properties

##### dimensions -> List[DimensionResult]

Returns all non-None dimension results.

```python
for dim in result.dimensions:
    print(f"{dim.dimension}: {dim.state.value}")
```

##### has_inconsistencies -> bool

True if any dimension found definite inconsistencies.

##### has_anomalies -> bool

True if any dimension flagged suspicious findings.

##### is_clean -> bool

True if all analysed dimensions are consistent or verified.

#### Methods

##### to_dict() -> Dict[str, Any]

Convert the analysis to a dictionary suitable for JSON serialisation.

##### to_json() -> str

Convert the analysis to a formatted JSON string.

```python
json_str = result.to_json()
with open("report.json", "w") as f:
    f.write(json_str)
```

---

### DimensionResult

The result of analysing a single forensic dimension.

```python
from wu.state import DimensionResult, DimensionState, Confidence, Evidence
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `dimension` | str | Name of the dimension (e.g., "metadata", "visual"). |
| `state` | DimensionState | Epistemic state of the finding. |
| `confidence` | Confidence | Confidence level in the finding. |
| `evidence` | List[Evidence] | Supporting evidence for the finding. |
| `methodology` | str | Description of the analytical method used. |
| `raw_data` | Dict | Additional structured data for debugging. |

#### Properties

| Property | Returns True When |
|----------|-------------------|
| `is_problematic` | State is INCONSISTENT, TAMPERED, or INVALID |
| `is_suspicious` | State is SUSPICIOUS |
| `is_clean` | State is CONSISTENT or VERIFIED |

---

## Enumerations

### DimensionState

Epistemic states for forensic findings. These states are designed for legal clarity and can be explained to a jury in plain language.

```python
from wu.state import DimensionState
```

| Value | Meaning | Legal Interpretation |
|-------|---------|---------------------|
| `CONSISTENT` | No anomalies detected | "We checked and found nothing wrong" |
| `INCONSISTENT` | Clear contradictions found | "X contradicts Y; this requires explanation" |
| `SUSPICIOUS` | Anomalies warrant investigation | "This is unusual and warrants further inquiry" |
| `UNCERTAIN` | Insufficient data for analysis | "We could not perform this analysis" |
| `VERIFIED` | Valid content credentials (C2PA) | "Cryptographic verification succeeded" |
| `TAMPERED` | Credentials present but file modified | "The file has been altered since signing" |
| `MISSING` | No credentials present | "No provenance chain exists" |
| `INVALID` | Credentials present but invalid | "The signature failed verification" |

### OverallAssessment

Aggregated assessment across all analysed dimensions.

```python
from wu.state import OverallAssessment
```

| Value | Condition |
|-------|-----------|
| `NO_ANOMALIES` | All dimensions consistent; no issues detected |
| `ANOMALIES_DETECTED` | One or more dimensions flagged suspicious |
| `INCONSISTENCIES_DETECTED` | One or more dimensions found inconsistencies |
| `INSUFFICIENT_DATA` | All dimensions returned uncertain |

### AuthenticityAssessment

Assessment states for authenticity burden mode, where the epistemic framing is inverted from "prove it is fake" to "prove it is authentic".

```python
from wu.state import AuthenticityAssessment
```

| Value | Meaning |
|-------|---------|
| `VERIFIED_AUTHENTIC` | Strong provenance chain with multiple verification sources |
| `LIKELY_AUTHENTIC` | Consistent across dimensions with partial verification |
| `UNVERIFIED` | No red flags but no positive verification either |
| `INSUFFICIENT_DATA` | Cannot assess authenticity |
| `AUTHENTICITY_COMPROMISED` | Evidence of tampering detected |

### Confidence

Confidence level in a finding.

```python
from wu.state import Confidence
```

| Value | Meaning |
|-------|---------|
| `HIGH` | Strong evidence supporting the finding |
| `MEDIUM` | Moderate evidence |
| `LOW` | Weak evidence; finding should be interpreted cautiously |
| `NA` | Not applicable (e.g., for UNCERTAIN state) |

---

## Supporting Classes

### Evidence

A single piece of evidence supporting a forensic finding.

```python
from wu.state import Evidence

evidence = Evidence(
    finding="Quantisation tables inconsistent",
    explanation="Primary and secondary tables suggest different sources",
    contradiction="Region A quality 85, Region B quality 72",
    citation="Farid, H. (2016). Photo Forensics. MIT Press.",
    timestamp="2024-01-15T14:32:00"
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `finding` | str | Brief description of what was found |
| `explanation` | str | Detailed explanation of the finding |
| `contradiction` | str | Specific contradictory evidence, if applicable |
| `citation` | str | Academic or standards citation |
| `timestamp` | str | Temporal evidence, if applicable |

### CorrelationWarning

Warning generated when findings across dimensions conflict.

```python
from wu.state import CorrelationWarning
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `severity` | str | "critical", "high", "medium", or "low" |
| `category` | str | Type of conflict (e.g., "device_mismatch") |
| `dimensions` | List[str] | Which dimensions conflict |
| `finding` | str | Human-readable description |
| `details` | Dict | Supporting data |

### AuthenticityResult

Result of authenticity burden analysis.

```python
from wu.state import AuthenticityResult
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `assessment` | AuthenticityAssessment | Overall authenticity assessment |
| `confidence` | float | Confidence score (0.0 to 1.0) |
| `verification_chain` | List[str] | Dimensions that positively verified |
| `gaps` | List[str] | Missing provenance |
| `summary` | str | Human-readable summary |

---

## Aggregation

### EpistemicAggregator

Combines dimension results into an overall assessment following epistemic asymmetry principles: a single inconsistency is significant, whilst consistency merely indicates absence of detected anomalies.

```python
from wu.aggregator import EpistemicAggregator

aggregator = EpistemicAggregator()
overall = aggregator.aggregate(dimension_results)
summary = aggregator.generate_summary(dimension_results)
corroboration = aggregator.generate_corroboration_summary(dimension_results)
```

### AuthenticityAggregator

Aggregates findings with authenticity burden (prove authenticity rather than detect manipulation).

```python
from wu.aggregator import AuthenticityAggregator

aggregator = AuthenticityAggregator()
result = aggregator.aggregate(dimension_results)
print(result.assessment)  # AuthenticityAssessment enum
print(result.confidence)  # 0.0 to 1.0
```

---

## Cross-Dimension Correlation

### DimensionCorrelator

Analyses relationships between dimension results to detect contradictions that individual dimensions might miss.

```python
from wu.correlator import DimensionCorrelator

correlator = DimensionCorrelator()
warnings = correlator.correlate(analysis)

for warning in warnings:
    print(f"[{warning.severity}] {warning.finding}")
```

The correlator checks for:

| Category | Description | Severity |
|----------|-------------|----------|
| `device_mismatch` | Metadata device conflicts with PRNU fingerprint | Critical |
| `c2pa_conflict` | C2PA verified but other dimensions show manipulation | Critical |
| `thumbnail_mismatch` | Thumbnail differs but no editing software in metadata | High |
| `lipsync_enf_conflict` | Lip-sync desync but ENF continuous | High |
| `temporal_impossibility` | Image created after it was digitised | High |
| `compression_conflict` | High quality claimed but double compression detected | Medium |
| `geometric_impossibility` | Lighting and shadow directions conflict | Medium |
| `enf_gps_mismatch` | GPS location conflicts with detected power grid frequency | Medium |
| `aigen_metadata_conflict` | AI generation detected but metadata claims camera capture | Medium |

---

## Command-Line Interface

Wu provides a command-line interface for direct use without Python scripting.

### analyze

Analyse a single media file.

```bash
wu analyze photo.jpg
wu analyze photo.jpg --json
wu analyze photo.jpg -o report.json
wu analyze photo.jpg --verbose
wu analyze photo.jpg --copymove --prnu --lighting
wu analyze photo.jpg --authenticity-mode
```

### batch

Analyse multiple files.

```bash
wu batch *.jpg
wu batch photos/*.png --output reports/
wu batch evidence/*.jpg --json
```

### report

Generate a court-ready PDF forensic report.

```bash
wu report photo.jpg
wu report photo.jpg -o report.pdf
wu report evidence.jpg --examiner "John Smith" --case "2024-001"
```

### formats

List supported file formats.

```bash
wu formats
```

### verify

Verify Wu installation against reference test vectors.

```bash
wu verify
wu verify --verbose
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No anomalies detected |
| 1 | Anomalies detected (suspicious findings) |
| 2 | Inconsistencies detected (definite problems) |

---

## Native Acceleration

Wu includes hand-written x86-64 AVX2 assembly kernels for performance-critical operations. These are loaded automatically when available and fall back to pure Python implementations otherwise.

### Available Kernels

| Kernel | Function | Approximate Speedup |
|--------|----------|---------------------|
| `copymove.asm` | Block matching for clone detection | ~20x |
| `prnu.asm` | Peak-to-Correlation Energy calculation | ~7x |
| `blockgrid.asm` | JPEG grid inconsistency detection | ~8x |
| `lighting.asm` | Light source direction estimation | ~6.5x |
| `h264_idct.asm` | Integer Inverse DCT for video | ~5x |
| `h264_inter.asm` | Motion compensation interpolation | ~4x |

### Building Native Extensions

Native extensions are built automatically during package installation if NASM is available:

```bash
# Windows
nasm -f win64 -o copymove.obj copymove.asm

# Linux
nasm -f elf64 -o copymove.o copymove.asm

# macOS
nasm -f macho64 -o copymove.o copymove.asm
```

The native module loader (`wu.native.simd`) handles platform detection and graceful fallback.

---

## Video Analysis

Wu includes native video forensics without dependency on FFmpeg for core analysis.

### VideoAnalyzer

```python
from wu.video.analyzer import VideoAnalyzer

analyzer = VideoAnalyzer()

# Iterate decoded frames
for frame in analyzer.iter_frames("video.mp4"):
    # frame is numpy array (H, W, 3) RGB
    pass

# Extract raw audio samples
for sample in analyzer.iter_audio_samples("video.mp4"):
    # sample is bytes
    pass
```

### Supported Codecs

| Container | Video Codecs | Audio Codecs |
|-----------|--------------|--------------|
| MP4/MOV | H.264 (Baseline/Main), MJPEG | AAC, MP3 |
| AVI | MJPEG | PCM, MP3 |
| MKV | H.264, MJPEG | AAC, MP3, FLAC |

### Bitstream Analysis

The video module performs forensic analysis at the codec level, examining:

- NAL unit structure and ordering
- Slice header consistency
- Motion vector distributions
- Quantisation parameter variations
- I-frame placement patterns

These markers can reveal splicing or re-encoding that container-level analysis would miss.

---

## Error Handling

Wu is designed to handle errors gracefully rather than raising exceptions during analysis. When an individual dimension fails, it produces a result with `UNCERTAIN` state and error details in the evidence list.

```python
result = analyzer.analyze("corrupt_file.jpg")

for dim in result.dimensions:
    if dim.state == DimensionState.UNCERTAIN:
        for evidence in dim.evidence:
            if "failed" in evidence.finding.lower():
                print(f"{dim.dimension}: {evidence.explanation}")
```

For batch analysis, failed files produce `WuAnalysis` objects with `INSUFFICIENT_DATA` assessment rather than interrupting the batch.

---

## Thread Safety

`WuAnalyzer` instances are thread-safe for concurrent `analyze()` calls. Each analysis creates independent dimension analyser instances and operates on separate file handles.

```python
from concurrent.futures import ThreadPoolExecutor

analyzer = WuAnalyzer()

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(analyzer.analyze, path) for path in file_paths]
    results = [f.result() for f in futures]
```

For maximum throughput on multi-core systems, use `analyze_batch()` with `parallel_files=True`, which provides both file-level and dimension-level parallelism.

---

## Extending Wu

### Adding a New Dimension

New analytical dimensions should follow the established pattern:

```python
from wu.state import DimensionResult, DimensionState, Confidence, Evidence

class CustomAnalyzer:
    """Docstring explaining the forensic methodology."""

    def analyze(self, file_path: str) -> DimensionResult:
        try:
            # Perform analysis
            findings = self._perform_analysis(file_path)

            return DimensionResult(
                dimension="custom",
                state=DimensionState.CONSISTENT,  # or SUSPICIOUS, INCONSISTENT
                confidence=Confidence.HIGH,
                evidence=[
                    Evidence(
                        finding="Brief finding description",
                        explanation="Detailed explanation",
                        citation="Academic reference if applicable"
                    )
                ],
                methodology="Description of method used",
                raw_data={"key": "value"}  # For debugging
            )
        except Exception as e:
            return DimensionResult(
                dimension="custom",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[
                    Evidence(
                        finding=f"Analysis failed: {type(e).__name__}",
                        explanation=str(e)
                    )
                ],
                methodology="Error during analysis"
            )
```

Register the dimension in `WuAnalyzer._build_analyzer_config()` to include it in standard analysis.

---

## References

- Daubert v. Merrell Dow Pharmaceuticals, 509 U.S. 579 (1993)
- Federal Rules of Evidence 702 (Expert Testimony)
- Farid, H. (2016). *Photo Forensics*. MIT Press.
- JEITA CP-3451C (Exif 2.32 specification)
- Scientific Working Group on Imaging Technology (SWGIT) guidelines
- C2PA Technical Specification (https://c2pa.org/specifications/)

---

*This document describes Wu version 1.5.x. API stability is maintained within major versions.*
