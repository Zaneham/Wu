# Wu - Epistemic Media Forensics Toolkit

Detects manipulated media with structural uncertainty output. Hopefully it will be suitable for court admissibility (Daubert standard).

Developed by **Zane Hambly**.

Named after **Chien-Shiung Wu** (1912-1997), who disproved parity conservation and found asymmetries everyone assumed didn't exist.

notes: 
video processing is actively being worked on. 
Wu does not explicitly target AI generated content however this tool does pick up AI generated images by proxy. AI generated or augmented videos and photos can leave traces. Please see the limitations and methodology documents for more information.


## Installation

```bash
pip install wu-forensics
```

## Quick Start

```bash
# Analyse a photo
wu analyze suspicious_photo.jpg

# JSON output
wu analyze photo.jpg --json

# Batch analysis
wu batch *.jpg --output reports/


```

## Python API

```python
from wu import WuAnalyzer

analyzer = WuAnalyzer()
result = analyzer.analyze("photo.jpg")

print(result.overall)  # OverallAssessment.NO_ANOMALIES
print(result.to_json())
```

## Detection Dimensions

Wu analyses images across multiple forensic dimensions:


| Dimension | What It Detects |
|-----------|-----------------|
| **metadata** | Device impossibilities, editing software, AI signatures, timestamp issues |
| **visual/ELA** | Error Level Analysis - compression inconsistencies from splicing |
| **quantization** | JPEG quality table mismatches between image regions |
| **copy-move** | Duplicated regions within the same image |
| **PRNU** | Photo Response Non-Uniformity - sensor fingerprint anomalies |
| **lighting** | Inconsistent light direction across image regions |
| **blockgrid** | JPEG block boundary misalignment |

## Benchmark Performance

Tested on standard forensic datasets (CASIA 2.0, CoMoFoD):

### CASIA 2.0 (Splice Forgeries)

| Dimension | Precision | Recall | FPR |
|-----------|-----------|--------|-----|
| **quantization** | **95%** | 39% | 2% |
| **visual/ELA** | **91%** | 41% | 4% |
| **prnu** | 67% | 6% | 3% |
| **copy-move** | 57% | 47% | 36% |
| **lighting** | 57% | 64% | 48% |

### Combined Detection

| Strategy | Precision | Recall | FPR | Use Case |
|----------|-----------|--------|-----|----------|
| ELA + Quantization | **91%** | 41% | **4%** | Conservative/Legal |
| All dimensions | 57% | **90%** | 67% | Screening |

**Key finding**: ELA + Quantization provides 91% precision with only 4% false positive rate on splice forgeries.

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
