#!/usr/bin/env python
"""
json_to_pdf.py

Render the two JSON reports (client + internal) into polished PDFs using
Jinja2 templates. If WeasyPrint is not installed, this script will still
render HTML files that you can open in your browser and "Print to PDF".

Usage (from your MMQB folder):
  python json_to_pdf.py --client_json "out\ADSJ_08-30-25-1wk_vs_08-23-25-1wk_client.json" \
                        --internal_json "out\ADSJ_08-30-25-1wk_vs_08-23-25-1wk_internal.json" \
                        --outdir "out"

Prereqs (one-time):
  pip install jinja2 weasyprint

Note: On Windows, if WeasyPrint PDF export fails due to missing system
libraries, the script will still generate HTML files. You can open them
and save to PDF via the browser (Ctrl+P → Save as PDF).
"""

import os, sys, json, argparse
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Try WeasyPrint for PDF output; if it fails, we fall back to HTML-only.
try:
    from weasyprint import HTML  # type: ignore
    HAVE_WEASY = True
except Exception:
    HAVE_WEASY = False

TEMPLATE_DIR = "templates"  # expects templates/client.html and templates/internal.html

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_outdir(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)

def render_html(template_name: str, data: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(template_name)
    return template.render(report=data)

def write_output(html_str: str, html_path: Path, pdf_path: Path | None):
    html_path.write_text(html_str, encoding="utf-8")
    if pdf_path and HAVE_WEASY:
        try:
            HTML(string=html_str, base_url=str(html_path.parent)).write_pdf(str(pdf_path))
            return True
        except Exception as e:
            print(f"[!] WeasyPrint PDF export failed: {e}")
            return False
    return False

def main():
    p = argparse.ArgumentParser(description="Render client and internal report JSON to PDF/HTML.")
    p.add_argument("--client_json", required=True, help="Path to *_client.json")
    p.add_argument("--internal_json", required=True, help="Path to *_internal.json")
    p.add_argument("--outdir", default="out", help="Output folder (default: out)")
    args = p.parse_args()

    ensure_outdir(args.outdir)

    client_data = load_json(args.client_json)
    internal_data = load_json(args.internal_json)

    # Guess a client code prefix for filenames
    def guess_prefix(pth: str):
        name = Path(pth).stem
        # everything up to first '_' is usually client code
        return name.split("_")[0] if "_" in name else "CLIENT"

    prefix = guess_prefix(args.client_json)

    # Render CLIENT
    client_html = render_html("client.html", client_data)
    client_html_path = Path(args.outdir) / f"{prefix}_client.html"
    client_pdf_path  = Path(args.outdir) / f"{prefix}_client.pdf"
    client_pdf_ok = write_output(client_html, client_html_path, client_pdf_path if HAVE_WEASY else None)

    # Render INTERNAL
    # Internal JSON from your pipeline contains both executive_summary and tables.
    internal_html = render_html("internal.html", internal_data)
    internal_html_path = Path(args.outdir) / f"{prefix}_internal.html"
    internal_pdf_path  = Path(args.outdir) / f"{prefix}_internal.pdf"
    internal_pdf_ok = write_output(internal_html, internal_html_path, internal_pdf_path if HAVE_WEASY else None)

    # Summary
    print("=== Render results ===")
    print(f"Client HTML : {client_html_path}")
    print(f"Client PDF  : {client_pdf_path if client_pdf_ok else '(WeasyPrint not available — open HTML and print to PDF)'}")
    print(f"Internal HTML: {internal_html_path}")
    print(f"Internal PDF : {internal_pdf_path if internal_pdf_ok else '(WeasyPrint not available — open HTML and print to PDF)'}")
    if not HAVE_WEASY:
        print("\nTip: To enable direct PDF export, install WeasyPrint:")
        print("     pip install weasyprint")
        print("If that fails on Windows, simply open the HTML files and use Ctrl+P → Save as PDF.")

if __name__ == "__main__":
    main()
