#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convierte report.md a report.html y luego report.pdf usando Chrome headless."""
import subprocess
import sys
from pathlib import Path
import markdown2

ROOT = Path(__file__).resolve().parent.parent
MD_FILE = ROOT / "results" / "report.md"
HTML_FILE = ROOT / "results" / "report.html"
PDF_FILE = ROOT / "results" / "report.pdf"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #1a1a2e;
    margin: 0;
    padding: 0;
    background: #fff;
}

/* ---- PAGE LAYOUT ---- */
@page {
    size: A4;
    margin: 18mm 15mm 18mm 15mm;
    @bottom-center {
        content: "TTS/STT Benchmark · Pág. " counter(page) " / " counter(pages);
        font-size: 8pt;
        color: #888;
    }
}

/* ---- HEADER COVER ---- */
h1 {
    font-size: 22pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 3px solid #6366f1;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 6px;
}

h2 {
    font-size: 14pt;
    font-weight: 700;
    color: #1e293b;
    border-left: 4px solid #6366f1;
    padding-left: 10px;
    margin-top: 28px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #334155;
    margin-top: 18px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

h4 {
    font-size: 10pt;
    font-weight: 600;
    color: #475569;
    margin-top: 12px;
    margin-bottom: 4px;
    page-break-after: avoid;
}

/* ---- TABLES ---- */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 16px 0;
    font-size: 8.5pt;
    page-break-inside: auto;
}

thead tr {
    background: #6366f1;
    color: #fff;
}

thead th {
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}

tbody tr:nth-child(even) { background: #f8fafc; }
tbody tr:nth-child(odd)  { background: #ffffff; }
tbody tr:hover           { background: #ede9fe; }

tbody td {
    padding: 5px 8px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

/* ---- CODE / INLINE CODE ---- */
code {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    padding: 1px 5px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 8pt;
    color: #7c3aed;
}

pre {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 8pt;
    overflow-x: auto;
    margin: 10px 0;
}

pre code {
    background: none;
    border: none;
    padding: 0;
    color: #e2e8f0;
    font-size: 8pt;
}

/* ---- BLOCKQUOTES ---- */
blockquote {
    border-left: 4px solid #a5b4fc;
    background: #eef2ff;
    margin: 10px 0;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 9pt;
    color: #3730a3;
}

blockquote p { margin: 0; }

/* ---- PARAGRAPHS & LISTS ---- */
p { margin: 6px 0 10px 0; }

ul, ol {
    padding-left: 20px;
    margin: 6px 0 10px 0;
}

li { margin-bottom: 3px; }

/* ---- HORIZONTAL RULE ---- */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 20px 0;
}

/* ---- STRONG / EM ---- */
strong { color: #0f172a; font-weight: 600; }
em     { color: #475569; }

/* ---- SECTION BREAKS ---- */
h2:not(:first-child) { page-break-before: auto; }

/* Sections 4 and 5 start on new page */
h2#4-ranking-comparativo-calidad-×-latencia-×-precio,
h2#5-conclusiones-para-presentación {
    page-break-before: always;
}

/* ---- FOOTER META ---- */
.meta {
    font-size: 8.5pt;
    color: #64748b;
    margin-bottom: 20px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 10px;
}
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width"/>
  <title>Reporte de Benchmark TTS / STT</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""

def main():
    print("Leyendo {}...".format(MD_FILE.name))
    md_text = MD_FILE.read_text(encoding="utf-8")

    # Convertir MD → HTML con soporte de tablas y extras
    html_body = markdown2.markdown(
        md_text,
        extras=["tables", "fenced-code-blocks", "strike", "header-ids",
                "smarty-pants", "break-on-newline"],
    )

    # Mover metadatos fuera del <h1> en una clase .meta
    html_body = html_body.replace(
        "<strong>Generado:</strong>",
        "</p><p class='meta'><strong>Generado:</strong>",
        1
    )

    full_html = HTML_TEMPLATE.format(css=CSS, body=html_body)
    HTML_FILE.write_text(full_html, encoding="utf-8")
    print("HTML generado -> {}".format(HTML_FILE))

    # Chrome headless → PDF
    chrome_cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--print-to-pdf=" + str(PDF_FILE),
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=5000",
        str(HTML_FILE),
    ]

    print("Generando PDF con Chrome headless...")
    result = subprocess.run(chrome_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error al generar PDF:")
        print(result.stderr)
        sys.exit(1)

    size_kb = PDF_FILE.stat().st_size // 1024
    print("PDF generado -> {}  ({} KB)".format(PDF_FILE, size_kb))


if __name__ == "__main__":
    main()
