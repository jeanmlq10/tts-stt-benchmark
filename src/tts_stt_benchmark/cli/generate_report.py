"""
CLI: generate_report

Usage
-----
generate_report --results_dir results/
generate_report --results_dir results/ --output results/report_2026.md
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from tts_stt_benchmark.reporting.report_builder import save_report

console = Console()


@click.command("generate_report")
@click.option(
    "--results_dir", "-r",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory containing tts/ and stt/ result subdirectories.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Output Markdown file path (default: <results_dir>/report.md).",
)
def main(results_dir: Path | None, output: Path | None):
    """Generate Markdown benchmark report from result JSON files."""
    from tts_stt_benchmark import config
    rd = results_dir or config.output_dir()
    out = save_report(rd, output)
    console.print(f"[green]Report written to {out}[/green]")
