# Wu - Epistemic Media Forensics Toolkit

Wu is a forensic toolkit designed to detect manipulated media by providing structured uncertainty outputs, a methodology developed to meet the rigorous requirements of court admissibility under the Daubert standard. This software is named in honour of **Chien-Shiung Wu** (1912-1997), a pioneering physicist whose meticulous experimental work disproved the principle of parity conservation and revealed fundamental asymmetries that had previously been assumed non-existent.

Developed by **Zane Hambly**, the toolkit provides a systematic framework for the technical examination of digital evidence across multiple modality-specific dimensions. Whilst the toolkit does not explicitly target wholly synthetic generative content, the forensic methodology employed frequently identifies anomalies in AI-augmented media through the detection of proxy technical inconsistencies, as further detailed in the associated limitations and methodology documentation.

## Installation

```bash
pip install wu-forensics
```

## Quick Start

```bash
# Analyse a photo or video file
wu analyze suspicious_media.mp4

# Generate a detailed JSON report for automated pipelines
wu analyze evidence.jpg --json

# Perform batch analysis on a directory of files
wu batch ./evidence/ --output reports/
```

## Detection Dimensions

Wu analyses media across multiple forensic dimensions to identify technical inconsistencies that may indicate manipulation:

| Dimension | Scope of Detection |
|-----------|--------------------|
| **metadata** | Analyses EXIF headers for device impossibilities, editing software signatures, and GPS consistency. |
| **visual/ELA** | Examines Error Level Analysis to detect compression inconsistencies typically arising from splicing. |
| **quantisation** | Identifies JPEG quality table mismatches across different regions of a single image. |
| **copy-move** | Detects duplicated pixel regions through block-based and keypoint-based matching algorithms. |
| **video** | Analyses native H.264/MJPEG bitstreams for container anomalies and codec-level splicing markers. |
| **audio** | Inspects Electric Network Frequency (ENF) continuity and spectral discontinuities in audio tracks. |
| **cross-modal** | Correlates findings between audio and video streams to identify temporal inconsistencies. |
| **prnu** | Computes Photo Response Non-Uniformity fingerprints to verify sensor-level consistency. |
| **lighting** | Evaluates the physical plausibility of light direction across various image components. |

## Benchmark Performance

Tested on standard forensic datasets (CASIA 2.0, CoMoFoD):

### CASIA 2.0 (Splice Forgeries)

| Dimension | Precision | Recall | FPR |
|-----------|-----------|--------|-----|
| **quantisation** | **95%** | 39% | 2% |
| **visual/ELA** | **91%** | 41% | 4% |
| **prnu** | 67% | 6% | 3% |
| **copy-move** | 57% | 47% | 36% |
| **lighting** | 57% | 64% | 48% |

### Combined Detection

| Strategy | Precision | Recall | FPR | Use Case |
|----------|-----------|--------|-----|----------|
| ELA + Quantisation | **91%** | 41% | **4%** | Conservative/Legal |
| All dimensions | 57% | **90%** | 67% | Screening |

**Key finding**: ELA + Quantisation provides 91% precision with only 4% false positive rate on splice forgeries.

### CoMoFoD (Copy-Move Forgeries)

Copy-move within the same image is harder to detect (identical compression/quality):

| Dimension | Precision | Recall | FPR |
|-----------|-----------|--------|-----|
| **prnu** | 61% | 38% | 24% |
| copy-move | 50% | 68% | 68% |

*Note: CoMoFoD includes "similar but genuine objects" designed to challenge detectors.*

## Epistemic States

Unlike binary classifiers, Wu reports structured uncertainty:

| State | Meaning |
|-------|---------|
| `CONSISTENT` | No anomalies detected (not proof of authenticity) |
| `INCONSISTENT` | Clear contradictions found |
| `SUSPICIOUS` | Anomalies that warrant investigation |
| `UNCERTAIN` | Insufficient data for analysis |

## Court Admissibility - in progress.

Wu is designed with the Daubert standard in mind:

1. **Testable methodology**: Every finding is reproducible
2. **Known error rates**: Confidence levels are explicit
3. **Peer review**: Academic citations throughout
4. **General acceptance**: Based on EXIF standards (JEITA CP-3451C)

## References

- Wu, C.S. et al. (1957). Experimental Test of Parity Conservation in Beta Decay. *Physical Review*, 105(4), 1413-1415.
- Farid, H. (2016). *Photo Forensics*. MIT Press.
- JEITA CP-3451C (Exif 2.32 specification)
- Daubert v. Merrell Dow Pharmaceuticals, 509 U.S. 579 (1993)
- Wen, B. et al. (2016). COVERAGE - A Novel Database for Copy-Move Forgery Detection. *IEEE ICIP*.
- Dong, J. et al. (2013). CASIA Image Tampering Detection Evaluation Database. *IEEE ChinaSIP*.

## AI Usage

This project uses Claude (Anthropic) to assist with summarising test results across 700+ test cases. All code, forensic methodology, and documentation are human-authored by me.



## License

MIT
