# Wu Methodology and Epistemic Framework

## Purpose

Wu is a forensic analysis toolkit developed by **Zane Hambly**, designed to identify inconsistencies in media files through multiple independent lines of technical inquiry. It is named after Chien-Shiung Wu, the physicist whose meticulous experimental work revealed asymmetries that prevailing theory assumed could not exist. In that spirit, Wu examines media files not to render verdicts but to surface evidence that warrants explanation.


## Epistemic Philosophy

Wu operates under a deliberately conservative epistemic framework that prioritises specificity over sensitivity. The system is designed to minimise false positives at the cost of potentially missing some manipulated files, because in forensic contexts an unfounded accusation of manipulation can be as harmful as failing to detect genuine tampering. When Wu reports an inconsistency, that finding reflects genuine technical anomalies in the file structure, metadata, or pixel content; when Wu reports no anomalies, this indicates only that the analyses performed did not detect inconsistencies using the methods available.

It is essential to understand what Wu does and does not claim. An absence of detected anomalies is not proof of authenticity; it means only that the specific technical examinations performed did not reveal inconsistencies. Conversely, the presence of anomalies does not constitute proof of malicious manipulation, as many legitimate workflows such as format conversion, batch processing, or standard editing operations can produce technical artefacts that appear anomalous under forensic examination. Wu identifies what requires explanation; the interpretation of that evidence in context remains the province of qualified human examiners.

The system expresses uncertainty explicitly rather than concealing it behind false confidence. Each dimension of analysis reports one of several epistemic states: consistent findings indicate no anomalies were detected, inconsistent findings indicate clear contradictions in the technical evidence, suspicious findings indicate anomalies that warrant further investigation, and uncertain findings indicate that the analysis could not be performed or the evidence was insufficient to draw conclusions. This structured uncertainty allows fact-finders to understand precisely what each examination did and did not establish.

## Multi-Dimensional Analysis

Wu examines media files across multiple independent dimensions, each representing a distinct technical domain with its own methodology and evidence base. The dimensions are designed to be orthogonal, meaning that the evidence gathered by one dimension neither depends upon nor duplicates the evidence gathered by another. This independence is forensically valuable because it allows for corroboration: when multiple unrelated analyses independently point toward the same conclusion, the combined evidence is substantially stronger than any single finding would be alone.

The current analytical framework includes metadata analysis examining EXIF timestamps, software signatures, and GPS coordinates for internal consistency; visual analysis employing error level analysis and compression artefact examination; thumbnail forensics comparing embedded preview images against the main image content; geometric analysis examining shadow directions and vanishing point consistency for physical plausibility; and compression forensics analysing JPEG quantisation tables and block grid alignment. For video evidence, the methodology extends into native H.264 and MJPEG bitstream analysis, where the system scrutinises codec-level parameters and motion compensation residuals to detect splicing markers that are typically lost during standard decoding.

A central component of the video forensic methodology is the use of SIMD-accelerated assembly kernels for core mathematical operations such as the Integer Inverse Transform (IDCT) and spatial interpolation filters. By re-implementing these critical paths in native assembly, Wu ensures bit-exact reproducibility and forensic integrity, allowing the system to verify that the decoded output has not been altered by the artefacts of third-party libraries. Furthermore, the toolkit employs a hybrid audio extraction strategy that natively identifies and locates bit-exact audio samples within ISO Base Media File Format (ISO BMFF) containers prior to forensic decoding. This approach ensures that the "raw" evidentiary data is isolated with surgical precision before undergoing subsequent spectral discontinuity and Electric Network Frequency (ENF) analysis.

When aggregating results across dimensions, Wu employs conservative logic that does not attempt to compute a single authenticity score or probability. Instead, it identifies the most concerning findings and presents them alongside the documented methodology, allowing human examiners to weigh the evidence appropriately. A finding of inconsistency in any single high-confidence dimension is sufficient to warrant concern, while consistent findings across all dimensions suggest only that no anomalies were detected using the specific methods available at the time of analysis.

## Corroboration and Convergent Evidence

When multiple independent dimensions produce findings that point toward the same conclusion, Wu notes this convergence as corroborating evidence. For example, if metadata analysis indicates the file was modified after its claimed creation date, quantisation table analysis reveals multiple compression passes, and thumbnail comparison shows the embedded preview differs from the main image, these three independent technical observations all suggest post-capture modification occurred. No single finding may be conclusive on its own, but their convergence strengthens the overall inference.

Wu does not assign numerical weights to corroborating evidence because the forensic significance of any finding depends heavily on context that the software cannot assess. The same technical anomaly might be entirely expected given one workflow and deeply suspicious given another. Wu surfaces the technical evidence and notes when findings converge; interpretation remains with qualified examiners who understand the full context of the matter at hand.

## Reproducibility and Transparency

All analyses performed by Wu are deterministic and reproducible. Given the same input file and the same version of Wu, the analysis will produce identical results. Each finding includes a methodology note explaining what technique was used, and where applicable, citations to the peer-reviewed literature establishing the scientific basis for that technique. This transparency allows opposing experts to understand precisely what was done, to reproduce the analysis independently, and to challenge specific methodological choices if warranted.

The software maintains a chain of custody record including a cryptographic hash of the analysed file, ensuring that any report can be definitively linked to the specific file examined. If questions arise about whether a file has been modified since analysis, the hash provides a definitive answer.

## Appropriate Use

Wu is designed to assist qualified forensic examiners, not to replace them. The software performs technical analyses and surfaces findings, but the interpretation of those findings in the context of a specific matter requires human expertise and judgement. Users should not represent Wu's output as definitive proof of authenticity or manipulation; rather, Wu's findings constitute technical evidence that may be relevant to such determinations when properly interpreted by qualified experts.

The software is appropriate for preliminary screening to identify files warranting detailed examination, for documenting the technical characteristics of media evidence, for identifying specific anomalies that require explanation, and for providing a structured framework for forensic reporting. It is not appropriate for rendering unsupervised verdicts about authenticity, for replacing the judgement of qualified human examiners, or for providing certainty where the evidence supports only provisional conclusions.
