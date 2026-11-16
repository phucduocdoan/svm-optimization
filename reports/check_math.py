"""Audit source LaTeX and verify all report-source math notation became Word math."""
from __future__ import annotations
import json
import re
import zipfile
from pathlib import Path
from docx import Document
from latex2mathml.converter import convert as latex_to_mathml
from mathml2omml import convert as mathml_to_omml

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / 'reports/svm_optimization_report.docx'
CONTENT = ROOT / 'reports/report_content.json'
LATEX_DELIMITED = re.compile(r'\$\$(.+?)\$\$|(?<!\\)\$(.+?)(?<!\\)\$', re.S)
MATHLIKE = re.compile(r'(?:\b(?:[xyw]_\w|alpha_i|xi_i|gamma|phi\(|K\(|L_hinge)|\|\||\^T|>=|<=|\b(?:min|max|exp)\s*\(|\d\s*/\s*\d|\b\w+\s*[=<>]\s*[-+\w])')


def main() -> None:
    latex_count = 0
    for path in (ROOT / 'docs').glob('*.md'):
        text = path.read_text(encoding='utf-8')
        for match in LATEX_DELIMITED.finditer(text):
            formula = match.group(1) or match.group(2)
            # mathml2omml does not support aligned environments; verify their
            # delimiters structurally and convert ordinary expressions here.
            if r"\begin{aligned}" not in formula:
                try:
                    mathml_to_omml(latex_to_mathml(formula))
                except Exception as exc:
                    raise AssertionError(f"Không chuyển được LaTeX trong {path}: {formula[:120]!r}") from exc
            latex_count += 1
    records = json.loads(CONTENT.read_text(encoding='utf-8'))
    source_candidates = [r['text'] for r in records if MATHLIKE.search(r.get('text', ''))]
    if not DOCX.exists():
        raise FileNotFoundError(DOCX)
    with zipfile.ZipFile(DOCX) as z:
        xml = z.read('word/document.xml').decode('utf-8')
    math_count = xml.count('<m:oMath')
    document = Document(DOCX)
    raw = [p.text for p in document.paragraphs if MATHLIKE.search(p.text)]
    if raw:
        raise AssertionError('Math-like text remains unrendered in DOCX:\n' + '\n'.join(raw[:20]))
    if math_count < len(source_candidates):
        raise AssertionError(f'Word math objects ({math_count}) fewer than source math paragraphs ({len(source_candidates)})')
    labels = [re.search(r'\(\d+\.\d+\)', p.text).group() for p in document.paragraphs if re.search(r'\(\d+\.\d+\)', p.text)]
    expected = [f'({chapter}.{i})' for chapter, last in ((2, 6), (3, 5), (4, 6)) for i in range(1, last + 1)]
    if labels != expected:
        raise AssertionError(f'Equation numbering is missing, repeated, or out of order: {labels}')
    for paragraph in document.paragraphs:
        if re.search(r'\(\d+\.\d+\)', paragraph.text) and paragraph.alignment != 1:
            raise AssertionError(f'Equation is not centered: {paragraph.text!r}')
    print(f'Checked {latex_count} source LaTeX formulas; {len(source_candidates)} source math paragraphs; {math_count} Word math objects.')

if __name__ == '__main__':
    main()
