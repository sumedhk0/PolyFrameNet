"""Render docs/submission/one_pager.md (+ its figure) to a one-page US-letter PDF via
headless Microsoft Edge. Usage: python scripts/render_one_pager.py [out.pdf]"""

import pathlib
import subprocess
import sys

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "submission" / "one_pager.md"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CSS = """
@page { size: letter; margin: 0.45in 0.5in; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 8.6pt; line-height: 1.28;
       color: #0b0b0b; }
h1 { font-size: 14pt; margin: 0 0 2pt 0; }
h2 { font-size: 10pt; margin: 6pt 0 2pt 0; color: #2a78d6; }
p, ul { margin: 2pt 0; }
li { margin: 1pt 0; }
ul { padding-left: 14pt; }
img { width: 100%; margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0; font-size: 8pt; }
th, td { border-bottom: 1px solid #e4e3df; padding: 1.5pt 4pt; text-align: left; }
th { color: #52514e; }
code { font-size: 7.8pt; }
"""

if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else MD.with_suffix(".pdf")
    body = markdown.markdown(MD.read_text(encoding="utf-8"), extensions=["tables"])
    html = MD.with_suffix(".html")
    html.write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{body}",
                    encoding="utf-8")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", html.as_uri()], check=True)
    html.unlink()
    print(out)
