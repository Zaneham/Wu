# Wu Limitations and Adversarial Considerations

## Scope of Detection

Wu analyses media files for technical inconsistencies that may indicate manipulation, but it is essential to understand the boundaries of what any forensic tool can detect. This document, compiled by **Zane**, describes what Wu does not detect, how a sophisticated adversary might attempt to evade detection, and why multi-dimensional analysis remains valuable despite these limitations. This transparency is not a weakness; forensic tools that do not acknowledge their limitations should be regarded with scepticism, and understanding what cannot be detected is as important as understanding what can.


## What Wu Does Not Detect

### Generative Content

Wu is designed to detect manipulation of existing photographs, not to identify wholly synthetic content. An image generated entirely by artificial intelligence or rendered from a three-dimensional model will not necessarily exhibit the technical inconsistencies that Wu examines, because such content was never a genuine photograph in the first place. The absence of compression artefacts, metadata anomalies, or geometric inconsistencies in AI-generated content does not indicate authenticity; it indicates only that the content was not created through the photographic process Wu assumes.

Detecting AI-generated imagery requires fundamentally different techniques, typically involving trained classifiers that identify statistical patterns characteristic of generative models. Such detection is an active research area with rapidly evolving capabilities on both the generation and detection sides, and it falls outside Wu's current scope. Users who need to assess whether content is AI-generated should employ specialised tools designed for that purpose.

### Skilled Manual Retouching

A careful human operator working with professional tools can perform manipulations that leave minimal technical traces. If someone uses raw image data, manually paints modifications at the pixel level, carefully manages compression to avoid telltale artefacts, and ensures metadata consistency throughout, the resulting file may exhibit no anomalies detectable through automated analysis. This does not mean such manipulation is undetectable in principle, only that it requires more intensive examination than automated tools can provide.

The practical significance of this limitation depends on context. Casual manipulation using common tools typically leaves abundant traces because users do not take such elaborate precautions. Sophisticated adversaries with substantial resources and expertise can potentially evade detection, but the effort required to do so comprehensively across all dimensions is substantial.

### Format-Specific Limitations

Certain analyses apply only to specific file formats. JPEG quantisation analysis requires JPEG files; examining a PNG that was never compressed as JPEG will yield uncertain results, not evidence of authenticity. EXIF thumbnail comparison requires files that contain embedded thumbnails, which not all capture devices produce. Electric network frequency analysis requires audio content captured in environments where mains hum is present. Users must understand which analyses are applicable to the evidence at hand and interpret uncertain results accordingly.

### Temporal Limitations

Wu examines the technical state of a file as it exists at the moment of analysis. If a file was manipulated and then subsequently processed in ways that overwrote the evidence of manipulation, Wu will examine only the final state. For example, if a manipulated JPEG is converted to PNG and back to JPEG, the new JPEG compression may obscure artefacts from the original manipulation. Wu cannot reconstruct the file's complete history; it can only examine what remains.

## Adversarial Evasion Techniques

Understanding how an adversary might attempt to evade detection illuminates both the limitations and the value of multi-dimensional analysis.

### Metadata Stripping and Reconstruction

The simplest evasion technique is to strip all metadata from a manipulated file, eliminating any inconsistencies in timestamps, software signatures, or GPS coordinates. More sophisticated adversaries might reconstruct plausible metadata that matches the claimed provenance. Wu will detect absent metadata and report this finding, but absent metadata is not conclusive evidence of manipulation since many legitimate workflows strip metadata for privacy or technical reasons. Reconstructed metadata may be internally consistent and thus pass metadata analysis, which is why metadata findings must be considered alongside other dimensions.

### Recompression from Uncompressed Source

If an adversary has access to uncompressed source material, they can perform manipulations at the raw pixel level and then compress the result as a fresh JPEG. This single-pass compression will not exhibit the quantisation artefacts and block grid anomalies that double compression produces. The resulting file may appear forensically clean in dimensions that examine compression history. However, other dimensions such as geometric consistency, lighting analysis, or content-level anomalies may still reveal problems that recompression cannot address.

### Careful Geometric Manipulation

An adversary aware of shadow and perspective analysis might take care to ensure that spliced elements have geometrically consistent lighting and vanishing points. This requires either substantial skill and effort or source material captured under matching conditions. While such care can defeat geometric analysis in isolation, maintaining consistency across all geometric constraints while also managing compression artefacts, metadata, and other dimensions simultaneously is considerably more difficult.

### Thumbnail Regeneration

Some tools allow regeneration of EXIF thumbnails after manipulation, eliminating the mismatch between thumbnail and main image that Wu examines. An adversary using such tools would evade thumbnail analysis specifically, but this evasion addresses only one dimension and does not affect the others.

## Why Multi-Dimensional Analysis Remains Valuable

The preceding sections might suggest that determined adversaries can evade any detection, but this conclusion misunderstands the practical value of multi-dimensional analysis. Each dimension an adversary must consider increases the complexity and effort required for successful evasion. Defeating metadata analysis requires attention to timestamps, software signatures, and structural consistency; defeating compression analysis requires understanding quantisation tables and block alignment; defeating geometric analysis requires physical plausibility of lighting and perspective; and so on through each dimension.

A sophisticated adversary might successfully address one or two dimensions while inadvertently leaving traces in others they did not consider. The more dimensions examined, the more opportunities exist to detect oversights. This is not a guarantee of detection, but it substantially raises the bar for undetected manipulation compared to any single-dimension analysis.

Furthermore, many real-world manipulation scenarios do not involve sophisticated adversaries taking elaborate precautions. Casual manipulation using consumer tools typically leaves abundant traces across multiple dimensions because the manipulator either does not know or does not care about forensic detectability. Wu is highly effective at identifying such manipulation, which constitutes the vast majority of cases encountered in practice.

## Implications for Forensic Practice

These limitations have several implications for appropriate use of Wu in forensic contexts.

First, Wu's findings should always be interpreted by qualified examiners who understand both the capabilities and limitations of the analyses performed. The software surfaces technical evidence; human expertise is required to assess its significance in context.

Second, the absence of detected anomalies should never be presented as proof of authenticity. Wu can identify inconsistencies but cannot verify that content is genuine. The most Wu can say is that specific examinations did not reveal problems, which is meaningfully different from asserting that no problems exist.

Third, findings should be reported with appropriate epistemic humility. When Wu detects an inconsistency, this indicates a technical anomaly requiring explanation, not necessarily malicious manipulation. When Wu detects nothing, this indicates only that the methods employed did not reveal inconsistencies, not that the file is necessarily authentic.

Fourth, understanding adversarial limitations helps calibrate confidence appropriately. Against casual manipulation, Wu's multi-dimensional analysis is highly effective. Against sophisticated adversaries with substantial resources and expertise, no automated tool provides guarantees, and intensive manual examination may be required.

## Ongoing Development

The landscape of media manipulation and forensic detection evolves continuously. New manipulation techniques emerge, and detection methods advance in response. Wu is designed to accommodate additional analysis dimensions as forensic research progresses, and users should ensure they are using current versions that incorporate the latest detection capabilities. The modular architecture allows new dimensions to be added without disrupting existing analyses, ensuring that the toolkit can grow to address emerging challenges while maintaining the reliability of established methods.
