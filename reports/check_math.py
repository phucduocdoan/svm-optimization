"""Validate LaTeX-to-Word math conversion independently of report generation."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from docx import Document
from latex2mathml.converter import convert as latex_to_mathml
from mathml2omml import convert as mathml_to_omml

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "reports" / "svm_optimization_report.docx"
INLINE = [r"x_i", r"y_i", r"w^T", r"x_i^T", r"\alpha_i", r"\xi_i", r"\phi(x_i)", r"\gamma", r"L_{\mathrm{hinge}}"]
DISPLAY = [r"\min_{w,b,\xi}\frac{1}{2}\|w\|^2+C\sum_i\xi_i", r"K(x_i,x_j)=\exp(-\gamma\|x_i-x_j\|^2)", r"\sum_i\alpha_i y_i=0"]


def main() -> None:
    for formula in INLINE + DISPLAY:
        mathml_to_omml(latex_to_mathml(formula))
    if not DOCX.exists():
        raise FileNotFoundError(f"Chưa có DOCX: {DOCX}")
    with zipfile.ZipFile(DOCX) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    math_count = xml.count("<m:oMath")
    if math_count < 30:
        raise AssertionError(f"Số công thức Word thấp bất thường: {math_count}")
    document = Document(DOCX)
    numbered = [p.text.strip() for p in document.paragraphs if re.fullmatch(r"\(\d+\.\d+\)", p.text.strip())]
    if not {"(2.4)", "(3.1)", "(4.1)"}.issubset(numbered):
        raise AssertionError(f"Thiếu số công thức theo chương: {numbered}")
    print(f"LaTeX conversion OK; {math_count} Word math objects; equation labels {', '.join(numbered)}")


if __name__ == "__main__":
    main()
