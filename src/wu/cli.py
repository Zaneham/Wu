"""
Wu command-line interface.

Usage:
    wu analyze photo.jpg
    wu analyze photo.jpg --json
    wu analyze photo.jpg -o report.json
    wu batch *.jpg --output results/

References:
    Click documentation: https://click.palletsprojects.com/
"""

import sys
import json
from pathlib import Path
from typing import Optional, List

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

from .analyzer import WuAnalyzer
from .state import OverallAssessment


# ANSI colors for terminal output
class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colorize(text: str, color: str, bold: bool = False) -> str:
    """Apply ANSI color to text."""
    prefix = Colors.BOLD if bold else ""
    return f"{prefix}{color}{text}{Colors.RESET}"


def format_assessment(assessment: OverallAssessment) -> str:
    """Format assessment with appropriate color."""
    if assessment == OverallAssessment.INCONSISTENCIES_DETECTED:
        return colorize("INCONSISTENCIES DETECTED", Colors.RED, bold=True)
    elif assessment == OverallAssessment.ANOMALIES_DETECTED:
        return colorize("ANOMALIES DETECTED", Colors.YELLOW, bold=True)
    elif assessment == OverallAssessment.NO_ANOMALIES:
        return colorize("NO ANOMALIES DETECTED", Colors.GREEN)
    else:
        return colorize("INSUFFICIENT DATA", Colors.BLUE)


def print_analysis(analysis, verbose: bool = False):
    """Print analysis results to terminal."""
    print()
    print(colorize("=" * 60, Colors.BLUE))
    print(colorize(" WU FORENSIC ANALYSIS", Colors.BLUE, bold=True))
    print(colorize("=" * 60, Colors.BLUE))
    print()

    # File info
    print(f"File: {analysis.file_path}")
    print(f"Hash: {analysis.file_hash[:16]}...")
    print(f"Analyzed: {analysis.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Overall assessment
    print(colorize("OVERALL ASSESSMENT:", Colors.BOLD))
    print(f"  {format_assessment(analysis.overall)}")
    print()

    # Findings summary
    if analysis.findings_summary:
        print(colorize("FINDINGS:", Colors.BOLD))
        for finding in analysis.findings_summary:
            if "METADATA" in finding and "Inconsist" in finding:
                print(f"  {colorize('!', Colors.RED)} {finding}")
            elif "Suspicious" in finding:
                print(f"  {colorize('?', Colors.YELLOW)} {finding}")
            else:
                print(f"  - {finding}")
        print()

    # Dimension details (verbose mode)
    if verbose:
        print(colorize("DIMENSION DETAILS:", Colors.BOLD))
        for dim in analysis.dimensions:
            state_color = (
                Colors.RED if dim.is_problematic else
                Colors.YELLOW if dim.is_suspicious else
                Colors.GREEN if dim.is_clean else
                Colors.BLUE
            )
            print(f"\n  [{dim.dimension.upper()}]")
            print(f"    State: {colorize(dim.state.value, state_color)}")
            print(f"    Confidence: {dim.confidence.value}")
            if dim.methodology:
                print(f"    Method: {dim.methodology}")
            for ev in dim.evidence:
                print(f"    - {ev.finding}")
                if ev.explanation:
                    print(f"      {ev.explanation}")
        print()

    print(colorize("=" * 60, Colors.BLUE))


if HAS_CLICK:
    @click.group()
    @click.version_option(version="1.2.0", prog_name="wu")
    def cli():
        """
        Wu - Epistemic Media Forensics Toolkit

        Analyzes media files for signs of manipulation with
        structured uncertainty output suitable for court use.

        Named after Chien-Shiung Wu (1912-1997), physicist who
        found asymmetries everyone assumed didn't exist.
        """
        pass

    @cli.command()
    @click.argument("file_path", type=click.Path(exists=True))
    @click.option("--json", "output_json", is_flag=True, help="Output as JSON")
    @click.option("-o", "--output", type=click.Path(), help="Write output to file")
    @click.option("-v", "--verbose", is_flag=True, help="Show detailed analysis")
    @click.option("--no-metadata", is_flag=True, help="Disable metadata analysis")
    @click.option("--no-c2pa", is_flag=True, help="Disable C2PA credential check")
    @click.option("--no-visual", is_flag=True, help="Disable visual analysis (ELA)")
    @click.option("--enf", is_flag=True, help="Enable ENF analysis (for audio/video)")
    @click.option("--copymove", is_flag=True, help="Enable copy-move (clone) detection")
    @click.option("--prnu", is_flag=True, help="Enable PRNU (sensor fingerprint) analysis")
    @click.option("--blockgrid", is_flag=True, help="Enable JPEG block grid analysis")
    @click.option("--lighting", is_flag=True, help="Enable lighting consistency analysis")
    @click.option("--audio", is_flag=True, help="Enable audio forensics (spectral, ENF, noise)")
    @click.option("--thumbnail", is_flag=True, help="Enable EXIF thumbnail mismatch detection")
    @click.option("--shadows", is_flag=True, help="Enable shadow direction consistency analysis")
    @click.option("--perspective", is_flag=True, help="Enable vanishing point consistency analysis")
    @click.option("--quantization", is_flag=True, help="Enable JPEG quantization table forensics")
    @click.option("--aigen", is_flag=True, help="Enable AI generation indicator analysis")
    @click.option("--video/--no-video", default=True, help="Enable/disable video and cross-modal analysis")
    def analyze(
        file_path: str,
        output_json: bool,
        output: Optional[str],
        verbose: bool,
        no_metadata: bool,
        no_c2pa: bool,
        no_visual: bool,
        enf: bool,
        copymove: bool,
        prnu: bool,
        blockgrid: bool,
        lighting: bool,
        audio: bool,
        thumbnail: bool,
        shadows: bool,
        perspective: bool,
        quantization: bool,
        aigen: bool,
        video: bool
    ):
        """
        Analyze a media file for manipulation.

        Examples:

            wu analyze photo.jpg

            wu analyze photo.jpg --json

            wu analyze photo.jpg -o report.json

            wu analyze photo.jpg --no-visual  (skip ELA analysis)

            wu analyze recording.wav --enf  (enable ENF timestamp verification)

            wu analyze photo.jpg --copymove  (detect cloned regions)

            wu analyze photo.jpg --prnu  (sensor fingerprint analysis)

            wu analyze photo.jpg --blockgrid  (JPEG block alignment)

            wu analyze photo.jpg --lighting  (lighting direction analysis)

            wu analyze recording.wav --audio  (audio spectral/ENF forensics)

            wu analyze photo.jpg --thumbnail  (EXIF thumbnail mismatch)

            wu analyze photo.jpg --shadows  (shadow direction consistency)

            wu analyze photo.jpg --perspective  (vanishing point consistency)

            wu analyze photo.jpg --quantization  (JPEG quantization table forensics)

            wu analyze photo.jpg --aigen  (AI generation indicator analysis)
        """
        analyzer = WuAnalyzer(
            enable_metadata=not no_metadata,
            enable_c2pa=not no_c2pa,
            enable_visual=not no_visual,
            enable_enf=enf,
            enable_copymove=copymove,
            enable_prnu=prnu,
            enable_blockgrid=blockgrid,
            enable_lighting=lighting,
            enable_audio=audio,
            enable_thumbnail=thumbnail,
            enable_shadows=shadows,
            enable_perspective=perspective,
            enable_quantization=quantization,
            enable_aigen=aigen,
            enable_video=video
        )

        if not analyzer.is_supported(file_path):
            click.echo(f"Warning: {file_path} may not be fully supported", err=True)

        result = analyzer.analyze(file_path)

        if output_json or output:
            json_output = result.to_json()



            if output:
                Path(output).write_text(json_output)
                click.echo(f"Analysis written to {output}")
            else:
                click.echo(json_output)
        else:
            print_analysis(result, verbose=verbose)

        # Exit code based on findings
        if result.overall == OverallAssessment.INCONSISTENCIES_DETECTED:
            sys.exit(2)
        elif result.overall == OverallAssessment.ANOMALIES_DETECTED:
            sys.exit(1)
        else:
            sys.exit(0)

    @cli.command()
    @click.argument("files", nargs=-1, type=click.Path(exists=True))
    @click.option("-o", "--output", type=click.Path(), help="Output directory for reports")
    @click.option("--json", "output_json", is_flag=True, help="Output as JSON")
    def batch(files: tuple, output: Optional[str], output_json: bool):
        """
        Analyze multiple files.

        Examples:

            wu batch *.jpg

            wu batch photos/*.png --output reports/
        """
        if not files:
            click.echo("No files specified", err=True)
            sys.exit(1)

        analyzer = WuAnalyzer()

        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)

        issues_found = 0
        for file_path in files:
            result = analyzer.analyze(file_path)

            if output:
                report_path = output_dir / f"{Path(file_path).stem}_wu.json"
                report_path.write_text(result.to_json())

            if output_json:
                click.echo(result.to_json())
            else:
                status = format_assessment(result.overall)
                click.echo(f"{file_path}: {status}")

            if result.overall in (
                OverallAssessment.INCONSISTENCIES_DETECTED,
                OverallAssessment.ANOMALIES_DETECTED
            ):
                issues_found += 1

        if output:
            click.echo(f"\nReports written to {output}/")

        click.echo(f"\nAnalyzed {len(files)} files, {issues_found} with issues")
        sys.exit(1 if issues_found > 0 else 0)

    @cli.command()
    @click.argument("file_path", type=click.Path(exists=True))
    @click.option("-o", "--output", type=click.Path(), help="Output PDF path")
    @click.option("--examiner", type=str, help="Examiner name for report")
    @click.option("--case", type=str, help="Case/matter number")
    @click.option("--notes", type=str, help="Additional examiner notes")
    def report(
        file_path: str,
        output: Optional[str],
        examiner: Optional[str],
        case: Optional[str],
        notes: Optional[str]
    ):
        """
        Generate a court-ready PDF forensic report.

        Examples:

            wu report photo.jpg

            wu report photo.jpg -o report.pdf

            wu report evidence.jpg --examiner "John Smith" --case "2024-001"
        """
        from .report import ForensicReportGenerator

        analyzer = WuAnalyzer()
        result = analyzer.analyze(file_path)

        # Default output path
        if not output:
            output = Path(file_path).stem + "_forensic_report.pdf"

        # Generate report
        try:
            generator = ForensicReportGenerator()
            generator.generate(
                result,
                output,
                examiner_name=examiner,
                case_number=case,
                notes=notes
            )
            click.echo(f"Forensic report generated: {output}")

            # Show summary
            status = format_assessment(result.overall)
            click.echo(f"Assessment: {status}")

        except ImportError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    @cli.command()
    def formats():
        """List supported file formats."""
        click.echo("Supported formats (Phase 0):\n")
        for fmt in WuAnalyzer.get_supported_formats():
            click.echo(f"  {fmt}")

    @cli.command()
    @click.option("-v", "--verbose", is_flag=True, help="Show detailed results")
    def verify(verbose: bool):
        """
        Verify Wu installation against reference test vectors.

        This command generates known test images, analyses them, and compares
        the results against expected values. Any discrepancy indicates potential
        modification or corruption of the Wu codebase.

        Use this to verify that your Wu installation is authentic and unmodified.

        Examples:

            wu verify

            wu verify --verbose
        """
        from .reference import verify_installation

        click.echo("Verifying Wu installation against reference vectors...")
        click.echo("")

        passed, report = verify_installation(verbose=verbose)

        click.echo(report)

        sys.exit(0 if passed else 1)

    def main():
        """Entry point for CLI."""
        # Ensure UTF-8 output even on Windows
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        cli()


else:
    def main():
        """Fallback when click is not installed."""
        print("Wu CLI requires 'click' package.")
        print("Install with: pip install click")
        sys.exit(1)


if __name__ == "__main__":
    main()
