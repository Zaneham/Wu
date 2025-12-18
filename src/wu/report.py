"""
Court-ready PDF report generator for Wu forensic analysis.

Generates professional forensic reports suitable for legal proceedings,
following best practices for expert witness documentation.

Report structure follows guidelines from:
- Scientific Working Group on Digital Evidence (SWGDE)
- Federal Rules of Evidence 702 (Expert Testimony)
- Daubert v. Merrell Dow Pharmaceuticals, 509 U.S. 579 (1993)

References:
    SWGDE Best Practices for Digital & Multimedia Evidence
    ASTM E2916-19 Standard Practice for Digital Evidence
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Image,
        HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from .state import WuAnalysis, DimensionState, OverallAssessment


class ForensicReportGenerator:
    """
    Generates court-ready PDF forensic reports.

    Reports include:
    - Executive summary
    - File identification and chain of custody info
    - Per-dimension analysis results
    - Evidence documentation
    - Methodology disclosure
    - Limitations and caveats

    Designed for Daubert standard compliance.
    """

    def __init__(self, pagesize=letter):
        """
        Initialize report generator.

        Args:
            pagesize: Page size (letter or A4)
        """
        if not HAS_REPORTLAB:
            raise ImportError(
                "reportlab is required for PDF generation. "
                "Install with: pip install reportlab"
            )

        self.pagesize = pagesize
        self.styles = self._create_styles()

    def _create_styles(self):
        """Create custom paragraph styles."""
        styles = getSampleStyleSheet()

        # Title style
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a1a2e'),
        ))

        # Section header
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#16213e'),
            borderPadding=5,
        ))

        # Subsection header
        styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#1a1a2e'),
        ))

        # Body text
        styles.add(ParagraphStyle(
            name='WuBodyText',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ))

        # Evidence text (monospace for hashes etc)
        styles.add(ParagraphStyle(
            name='Evidence',
            parent=styles['Code'],
            fontSize=9,
            leading=12,
            backColor=colors.HexColor('#f5f5f5'),
            borderPadding=8,
            spaceAfter=10,
        ))

        # Finding - positive
        styles.add(ParagraphStyle(
            name='FindingClean',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#0d7377'),
            leftIndent=20,
            spaceAfter=5,
        ))

        # Finding - problematic
        styles.add(ParagraphStyle(
            name='FindingProblem',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#c70039'),
            leftIndent=20,
            spaceAfter=5,
        ))

        # Finding - suspicious
        styles.add(ParagraphStyle(
            name='FindingSuspicious',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#ff8c00'),
            leftIndent=20,
            spaceAfter=5,
        ))

        # Disclaimer
        styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ))

        return styles

    def generate(
        self,
        analysis: WuAnalysis,
        output_path: str,
        examiner_name: Optional[str] = None,
        case_number: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Generate a forensic report PDF.

        Args:
            analysis: WuAnalysis result to document
            output_path: Path for output PDF
            examiner_name: Name of forensic examiner
            case_number: Case or matter number
            notes: Additional examiner notes

        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.pagesize,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )

        story = []

        # Title page content
        story.extend(self._build_header(analysis, case_number))
        story.append(Spacer(1, 0.3*inch))

        # Executive summary
        story.extend(self._build_executive_summary(analysis))
        story.append(Spacer(1, 0.2*inch))

        # File identification
        story.extend(self._build_file_identification(analysis))
        story.append(Spacer(1, 0.2*inch))

        # Analysis results
        story.extend(self._build_analysis_results(analysis))
        story.append(Spacer(1, 0.2*inch))

        # Track section number for dynamic numbering
        section_num = 4

        # Corroboration (if present)
        if analysis.corroboration_summary:
            story.extend(self._build_corroboration(analysis, section_num))
            story.append(Spacer(1, 0.2*inch))
            section_num += 1

        # Detailed findings
        story.extend(self._build_detailed_findings(analysis, section_num))
        story.append(Spacer(1, 0.2*inch))
        section_num += 1

        # Methodology
        story.extend(self._build_methodology(section_num))
        story.append(Spacer(1, 0.2*inch))
        section_num += 1

        # Limitations
        story.extend(self._build_limitations(section_num))
        story.append(Spacer(1, 0.2*inch))
        section_num += 1

        # Examiner information
        if examiner_name or notes:
            story.extend(self._build_examiner_section(examiner_name, notes, section_num))
            story.append(Spacer(1, 0.2*inch))

        # Footer/disclaimer
        story.extend(self._build_footer(analysis))

        # Build PDF
        doc.build(story)
        return output_path

    def _build_header(
        self,
        analysis: WuAnalysis,
        case_number: Optional[str]
    ) -> List:
        """Build report header."""
        elements = []

        # Title
        elements.append(Paragraph(
            "FORENSIC MEDIA ANALYSIS REPORT",
            self.styles['ReportTitle']
        ))

        # Horizontal rule
        elements.append(HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor('#1a1a2e'),
            spaceAfter=20,
        ))

        # Report metadata table
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        analysis_date = analysis.analyzed_at.strftime("%Y-%m-%d %H:%M:%S")

        meta_data = [
            ["Report Generated:", report_date],
            ["Analysis Performed:", analysis_date],
            ["Wu Version:", analysis.wu_version],
        ]
        if case_number:
            meta_data.insert(0, ["Case Number:", case_number])

        meta_table = Table(meta_data, colWidths=[1.5*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(meta_table)

        return elements

    def _build_executive_summary(self, analysis: WuAnalysis) -> List:
        """Build executive summary section."""
        elements = []

        elements.append(Paragraph(
            "1. EXECUTIVE SUMMARY",
            self.styles['SectionHeader']
        ))

        # Overall assessment with color coding
        assessment_text = self._format_assessment(analysis.overall)
        elements.append(Paragraph(
            f"<b>Overall Assessment:</b> {assessment_text}",
            self.styles['WuBodyText']
        ))

        # Summary of findings
        if analysis.findings_summary:
            elements.append(Paragraph(
                "<b>Key Findings:</b>",
                self.styles['WuBodyText']
            ))
            for finding in analysis.findings_summary:
                style = self._get_finding_style(finding)
                elements.append(Paragraph(f"• {finding}", style))
        else:
            elements.append(Paragraph(
                "No significant findings to report.",
                self.styles['WuBodyText']
            ))

        return elements

    def _build_file_identification(self, analysis: WuAnalysis) -> List:
        """Build file identification section."""
        elements = []

        elements.append(Paragraph(
            "2. FILE IDENTIFICATION",
            self.styles['SectionHeader']
        ))

        elements.append(Paragraph(
            "The following file was submitted for forensic analysis:",
            self.styles['WuBodyText']
        ))

        # File details table
        file_data = [
            ["File Path:", analysis.file_path],
            ["SHA-256 Hash:", analysis.file_hash],
            ["Analysis Timestamp:", analysis.analyzed_at.isoformat()],
        ]

        file_table = Table(file_data, colWidths=[1.5*inch, 5*inch])
        file_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Courier'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('BACKGROUND', (1, 1), (1, 1), colors.HexColor('#f5f5f5')),
        ]))
        elements.append(file_table)

        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(
            "<i>The SHA-256 hash provides a unique digital fingerprint of the analyzed file. "
            "Any modification to the file would result in a different hash value.</i>",
            self.styles['Disclaimer']
        ))

        return elements

    def _build_analysis_results(self, analysis: WuAnalysis) -> List:
        """Build analysis results section."""
        elements = []

        elements.append(Paragraph(
            "3. ANALYSIS RESULTS",
            self.styles['SectionHeader']
        ))

        if not analysis.dimensions:
            elements.append(Paragraph(
                "No analysis dimensions were available for this file.",
                self.styles['WuBodyText']
            ))
            return elements

        # Results table
        table_data = [["Dimension", "State", "Confidence", "Methodology"]]

        for dim in analysis.dimensions:
            state_text = self._format_state(dim.state)
            table_data.append([
                dim.dimension.title(),
                state_text,
                dim.confidence.value.upper(),
                dim.methodology or "N/A"
            ])

        results_table = Table(
            table_data,
            colWidths=[1.2*inch, 1.5*inch, 1*inch, 2.8*inch]
        )
        results_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            ('ALIGN', (3, 1), (3, -1), 'LEFT'),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(results_table)

        return elements

    def _build_corroboration(self, analysis: WuAnalysis, section_num: int = 4) -> List:
        """Build corroborating evidence section."""
        elements = []

        elements.append(Paragraph(
            f"{section_num}. CORROBORATING EVIDENCE",
            self.styles['SectionHeader']
        ))

        elements.append(Paragraph(
            "When multiple independent analytical dimensions identify concerns that "
            "point toward the same conclusion, this convergent evidence is forensically "
            "significant. Each dimension examines different technical properties of the "
            "file; independent agreement strengthens the overall inference substantially.",
            self.styles['WuBodyText']
        ))

        # Split the corroboration summary into paragraphs
        if analysis.corroboration_summary:
            paragraphs = analysis.corroboration_summary.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Check if this is a finding header
                    if para.strip().startswith('Finding'):
                        elements.append(Paragraph(
                            f"<b>{para.strip()}</b>",
                            self.styles['WuBodyText']
                        ))
                    else:
                        elements.append(Paragraph(
                            para.strip(),
                            self.styles['WuBodyText']
                        ))

        return elements

    def _build_detailed_findings(self, analysis: WuAnalysis, section_num: int = 4) -> List:
        """Build detailed findings section."""
        elements = []

        elements.append(Paragraph(
            f"{section_num}. DETAILED FINDINGS",
            self.styles['SectionHeader']
        ))

        for dim in analysis.dimensions:
            elements.append(Paragraph(
                f"{section_num}.{analysis.dimensions.index(dim)+1} {dim.dimension.title()} Analysis",
                self.styles['SubsectionHeader']
            ))

            elements.append(Paragraph(
                f"<b>State:</b> {self._format_state(dim.state)} | "
                f"<b>Confidence:</b> {dim.confidence.value.upper()}",
                self.styles['WuBodyText']
            ))

            if dim.evidence:
                for ev in dim.evidence:
                    # Finding
                    style = self._get_finding_style_for_state(dim.state)
                    elements.append(Paragraph(
                        f"<b>Finding:</b> {ev.finding}",
                        style
                    ))

                    # Explanation
                    if ev.explanation:
                        elements.append(Paragraph(
                            f"<i>Explanation:</i> {ev.explanation}",
                            self.styles['WuBodyText']
                        ))

                    # Contradiction
                    if ev.contradiction:
                        elements.append(Paragraph(
                            f"<i>Contradiction:</i> {ev.contradiction}",
                            self.styles['FindingProblem']
                        ))

                    # Citation
                    if ev.citation:
                        elements.append(Paragraph(
                            f"<i>Reference:</i> {ev.citation}",
                            self.styles['Disclaimer']
                        ))

                    elements.append(Spacer(1, 0.1*inch))
            else:
                elements.append(Paragraph(
                    "No specific findings recorded for this dimension.",
                    self.styles['WuBodyText']
                ))

        return elements

    def _build_methodology(self, section_num: int = 5) -> List:
        """Build methodology section."""
        elements = []

        elements.append(Paragraph(
            f"{section_num}. METHODOLOGY",
            self.styles['SectionHeader']
        ))

        # Epistemic philosophy
        elements.append(Paragraph(
            f"<b>{section_num}.1 Epistemic Framework</b>",
            self.styles['SubsectionHeader']
        ))

        elements.append(Paragraph(
            "Wu prioritises specificity over sensitivity. In the context of forensic "
            "analysis for legal proceedings, a false positive (incorrectly claiming "
            "manipulation) is more damaging than a false negative (failing to detect "
            "manipulation). The tool is therefore calibrated to minimise false positives, "
            "accepting that some manipulations will go undetected.",
            self.styles['WuBodyText']
        ))

        elements.append(Paragraph(
            "This approach reflects the asymmetry inherent in forensic reasoning: while "
            "a single definitive inconsistency can establish manipulation, no amount of "
            "consistent findings can prove authenticity. Absence of detected anomalies "
            "means only that no anomalies were found, not that none exist.",
            self.styles['WuBodyText']
        ))

        elements.append(Paragraph(
            f"<b>{section_num}.2 Multi-Dimensional Analysis</b>",
            self.styles['SubsectionHeader']
        ))

        elements.append(Paragraph(
            "Wu examines media files across multiple independent dimensions. Each dimension "
            "analyses a different technical aspect of the file. When multiple dimensions "
            "independently identify concerns pointing toward the same conclusion, this "
            "convergent evidence is forensically significant because it is unlikely to "
            "occur by chance.",
            self.styles['WuBodyText']
        ))

        elements.append(Paragraph(
            f"<b>{section_num}.3 Analytical Methods</b>",
            self.styles['SubsectionHeader']
        ))

        methodologies = [
            "<b>Metadata Analysis:</b> Extraction and verification of EXIF metadata "
            "per JEITA CP-3451C (Exif 2.32 specification). Device capabilities are "
            "cross-referenced against manufacturer specifications to identify "
            "impossible claims (for example, resolution exceeding sensor capability).",

            "<b>Software Signature Detection:</b> Identification of editing software "
            "signatures in metadata fields including Software, ProcessingSoftware, "
            "CreatorTool, and HistorySoftwareAgent.",

            "<b>AI Generation Detection:</b> Detection of AI image and video generation "
            "signatures including DALL-E, Midjourney, Stable Diffusion, and others. "
            "Includes detection of generation parameters embedded in metadata.",

            "<b>Timestamp Verification:</b> Analysis of temporal metadata for "
            "logical inconsistencies such as modification dates preceding capture "
            "dates or future timestamps.",

            "<b>Visual Forensics:</b> Error Level Analysis (ELA) and JPEG artifact "
            "examination to identify regions that may have been modified after "
            "initial compression.",

            "<b>JPEG Compression Analysis:</b> Quantization table forensics and block "
            "grid alignment analysis to detect evidence of recompression or splicing.",

            "<b>Geometric Consistency:</b> Shadow direction, lighting direction, and "
            "vanishing point analysis to identify physically impossible scene geometry.",

            "<b>Copy-Move Detection:</b> Identification of duplicated regions within "
            "an image that may indicate cloning or content duplication.",

            "<b>C2PA Verification:</b> Validation of Content Credentials (C2PA) "
            "provenance data when present, detecting tampering or invalid signatures.",
        ]

        for method in methodologies:
            elements.append(Paragraph(f"• {method}", self.styles['WuBodyText']))

        return elements

    def _build_limitations(self, section_num: int = 6) -> List:
        """Build limitations section."""
        elements = []

        elements.append(Paragraph(
            f"{section_num}. LIMITATIONS AND CAVEATS",
            self.styles['SectionHeader']
        ))

        elements.append(Paragraph(
            f"<b>{section_num}.1 Fundamental Limitations</b>",
            self.styles['SubsectionHeader']
        ))

        fundamental = [
            "A finding of 'No Anomalies Detected' indicates that this analysis did not "
            "identify signs of manipulation. It does NOT constitute proof of authenticity. "
            "Absence of evidence is not evidence of absence.",

            "This analysis examines only the dimensions implemented in the current version "
            "of Wu. Sophisticated manipulation techniques may exist that are not detectable "
            "by the methods employed.",

            "This tool is designed to assist forensic examination, not replace expert "
            "judgement. Results should be interpreted by qualified forensic examiners.",
        ]

        for limitation in fundamental:
            elements.append(Paragraph(f"• {limitation}", self.styles['WuBodyText']))

        elements.append(Paragraph(
            f"<b>{section_num}.2 What Wu Does Not Detect</b>",
            self.styles['SubsectionHeader']
        ))

        not_detected = [
            "<b>AI-generated content without metadata signatures:</b> Modern generative "
            "models can produce photorealistic images with no embedded signatures. Wu's "
            "metadata analysis will not identify such content unless the generation tool "
            "left identifiable traces.",

            "<b>Skilled manual retouching:</b> Expert-level manipulation using professional "
            "tools, when performed with attention to forensic artefacts, may leave no "
            "detectable traces in the dimensions Wu analyses.",

            "<b>Pre-capture staging:</b> Wu analyses the digital file, not the scene it "
            "depicts. A photograph of a staged scene is technically authentic even if "
            "the scene itself was constructed to deceive.",

            "<b>Semantic manipulation:</b> Cropping, selective framing, or misleading "
            "captions do not alter pixel data and are not detectable through technical "
            "forensics alone.",
        ]

        for item in not_detected:
            elements.append(Paragraph(f"• {item}", self.styles['WuBodyText']))

        elements.append(Paragraph(
            f"<b>{section_num}.3 Known Evasion Techniques</b>",
            self.styles['SubsectionHeader']
        ))

        elements.append(Paragraph(
            "An adversary aware of forensic methods may employ countermeasures. "
            "The following techniques can reduce or eliminate forensic traces:",
            self.styles['WuBodyText']
        ))

        evasion = [
            "<b>Metadata stripping:</b> Complete removal of EXIF and XMP data eliminates "
            "metadata-based analysis but may itself be suspicious if the purported source "
            "would normally include such data.",

            "<b>Recompression and format conversion:</b> Re-saving an image at different "
            "quality settings or converting between formats can obscure compression "
            "history and ELA artefacts.",

            "<b>Anti-forensic filters:</b> Deliberate addition of noise, slight blurring, "
            "or geometric transforms can disrupt copy-move detection and block grid analysis.",

            "<b>Metadata forgery:</b> Device identifiers and timestamps can be manually "
            "inserted to match a target narrative, though cross-referencing with device "
            "capabilities may reveal inconsistencies.",
        ]

        for item in evasion:
            elements.append(Paragraph(f"• {item}", self.styles['WuBodyText']))

        elements.append(Paragraph(
            f"<b>{section_num}.4 Data Limitations</b>",
            self.styles['SubsectionHeader']
        ))

        data_limits = [
            "Device capability databases may not include all device variants or may have "
            "minor specification inaccuracies. Unknown devices cannot be verified against "
            "manufacturer specifications.",

            "Metadata can be deliberately modified or stripped. The absence of metadata "
            "anomalies does not guarantee the file content is unaltered.",

            "Some analyses require minimum image quality or size. Very small images or "
            "heavily compressed files may yield uncertain results.",
        ]

        for item in data_limits:
            elements.append(Paragraph(f"• {item}", self.styles['WuBodyText']))

        return elements

    def _build_examiner_section(
        self,
        examiner_name: Optional[str],
        notes: Optional[str],
        section_num: int = 7
    ) -> List:
        """Build examiner section."""
        elements = []

        elements.append(Paragraph(
            f"{section_num}. EXAMINER INFORMATION",
            self.styles['SectionHeader']
        ))

        if examiner_name:
            elements.append(Paragraph(
                f"<b>Examining Party:</b> {examiner_name}",
                self.styles['WuBodyText']
            ))

        if notes:
            elements.append(Paragraph(
                "<b>Examiner Notes:</b>",
                self.styles['WuBodyText']
            ))
            elements.append(Paragraph(notes, self.styles['WuBodyText']))

        return elements

    def _build_footer(self, analysis: WuAnalysis) -> List:
        """Build report footer."""
        elements = []

        elements.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.gray,
            spaceBefore=20,
            spaceAfter=10,
        ))

        elements.append(Paragraph(
            f"Report generated by Wu Epistemic Media Forensics Toolkit v{analysis.wu_version}",
            self.styles['Disclaimer']
        ))

        elements.append(Paragraph(
            "Wu is named after Chien-Shiung Wu (1912-1997), physicist who found "
            "asymmetries everyone assumed didn't exist.",
            self.styles['Disclaimer']
        ))

        elements.append(Paragraph(
            "This report is provided for informational purposes. The software and this "
            "report are provided 'as is' without warranty of any kind.",
            self.styles['Disclaimer']
        ))

        return elements

    def _format_assessment(self, assessment: OverallAssessment) -> str:
        """Format overall assessment for display."""
        mapping = {
            OverallAssessment.NO_ANOMALIES: "NO ANOMALIES DETECTED",
            OverallAssessment.ANOMALIES_DETECTED: "ANOMALIES DETECTED - Further investigation recommended",
            OverallAssessment.INCONSISTENCIES_DETECTED: "INCONSISTENCIES DETECTED - Evidence of manipulation",
            OverallAssessment.INSUFFICIENT_DATA: "INSUFFICIENT DATA - Unable to make determination",
        }
        return mapping.get(assessment, str(assessment.value))

    def _format_state(self, state: DimensionState) -> str:
        """Format dimension state for display."""
        mapping = {
            DimensionState.CONSISTENT: "Consistent",
            DimensionState.INCONSISTENT: "INCONSISTENT",
            DimensionState.SUSPICIOUS: "Suspicious",
            DimensionState.UNCERTAIN: "Uncertain",
            DimensionState.VERIFIED: "Verified",
            DimensionState.TAMPERED: "TAMPERED",
            DimensionState.MISSING: "Missing",
            DimensionState.INVALID: "INVALID",
        }
        return mapping.get(state, state.value)

    def _get_finding_style(self, finding: str) -> ParagraphStyle:
        """Get appropriate style for a finding based on content."""
        finding_lower = finding.lower()
        if "inconsist" in finding_lower or "impossible" in finding_lower:
            return self.styles['FindingProblem']
        elif "suspicious" in finding_lower or "detected" in finding_lower:
            return self.styles['FindingSuspicious']
        else:
            return self.styles['FindingClean']

    def _get_finding_style_for_state(self, state: DimensionState) -> ParagraphStyle:
        """Get appropriate style based on dimension state."""
        if state in (DimensionState.INCONSISTENT, DimensionState.TAMPERED, DimensionState.INVALID):
            return self.styles['FindingProblem']
        elif state == DimensionState.SUSPICIOUS:
            return self.styles['FindingSuspicious']
        else:
            return self.styles['FindingClean']


def generate_report(
    analysis: WuAnalysis,
    output_path: str,
    **kwargs
) -> str:
    """
    Convenience function to generate a forensic report.

    Args:
        analysis: WuAnalysis result
        output_path: Path for output PDF
        **kwargs: Additional arguments for ForensicReportGenerator.generate()

    Returns:
        Path to generated PDF
    """
    generator = ForensicReportGenerator()
    return generator.generate(analysis, output_path, **kwargs)
