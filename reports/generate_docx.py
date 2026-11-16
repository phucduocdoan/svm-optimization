r"""Sinh báo cáo Word theo file định dạng của cô Xuân.

Chạy sau khi đã chạy benchmark:

    uv run python reports/generate_docx.py

Dùng report_base.docx làm nội dung nền, giữ nguyên các chương của nhóm.
Có thể cung cấp template khác bằng biến môi trường ``REPORT_TEMPLATE``.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from latex2mathml.converter import convert as latex_to_mathml
from mathml2omml import convert as mathml_to_omml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "svm_optimization_report.docx"
TEMPLATE = Path(os.environ.get("REPORT_TEMPLATE", ROOT / "reports" / "report_layout.docx"))
CONTENT = ROOT / "reports" / "report_content.json"
DATA = ROOT / "reports" / "data"
FIGURES = ROOT / "reports" / "figures"
ILLUSTRATIONS = ROOT / "docs" / "images" / "scientific"
DIAGRAMS = ROOT / "docs" / "diagrams"


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(str(text))
    run.bold = bold
    p.paragraph_format.space_after = Pt(0)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_field(paragraph, instruction: str) -> None:
    """Thêm field Word, để Word cập nhật số trang mục lục khi mở."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "Chọn mục lục và nhấn F9 trong Word để cập nhật."
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, placeholder, end])


def add_caption(document: Document, text: str) -> None:
    p = document.add_paragraph(style="Caption" if "Caption" in [s.name for s in document.styles] else "Content")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(text)


def add_figure(document: Document, path: Path, caption: str, width: float = 6.1) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Thiếu hình: {path}")
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(document, caption)


def add_table(document: Document, headers, rows, widths=None):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], header, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value)
    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    document.add_paragraph()
    return table


def add_equation(document: Document, latex: str, number: str) -> None:
    """Chèn công thức LaTeX thành Word OMML, có số công thức bên phải."""
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.tab_stops.add_tab_stop(Cm(8), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(16), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    mathml = latex_to_mathml(latex)
    omml = mathml_to_omml(mathml).replace(
        "<m:oMath>",
        '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">',
        1,
    )
    p._p.append(parse_xml(omml.encode("utf-8")))
    if number:
        p.add_run(f"\t{number}")


INLINE_MATH = re.compile(
    r"K\(x_i, x_j\)|phi\(x_i\)|phi\(x\)|x_i\^T|w\^T|"
    r"alpha_i|xi_i|L_hinge|gamma|[wxy]_[0-9]+|[wxy]_[a-z]|x_i|y_i|(?<!\w)C(?!\w)|"
    r"\|\|w\|\||\|\|x_i - x_j\|\||\|\|x_i\|\||"
    r">=|<=|=>|(?<=\s)\.(?=\s)|(?<!\w)[=+*/^<>-](?!\w)"
)
ATOM = r"(?:\|\|[^|]+\|\||\[[0-9.,\s+\-]+\](?:\^T)?|[A-Za-z]+(?:\^[A-Za-z]+)?(?:_[A-Za-z0-9]+)?(?:\([^()]*\))?|[+-]?\d+(?:\.\d+)?)"
MATH_EXPRESSION = re.compile(rf"(?<!\w){ATOM}(?:\s*(?:>=|<=|=>|=|>|<|\+|-|\*|/|\^|\.|≈)\s*{ATOM})+(?!\w)")


def to_latex(expression: str) -> str:
    expression = expression.strip().replace("||", r"\|")
    for old, new in [("alpha_i", r"\alpha_i"), ("xi_i", r"\xi_i"), ("gamma", r"\gamma"), ("phi", r"\phi"), ("L_hinge", r"L_{\mathrm{hinge}}"), (">=", r"\geq"), ("<=", r"\leq"), ("=>", r"\Rightarrow"), ("≈", r"\approx"), (" . ", r" \cdot ")]:
        expression = expression.replace(old, new)
    return expression


def format_inline_math(document: Document) -> None:
    """Render common SVM variables as native Word math without changing prose."""
    for paragraph in list(document.paragraphs):
        source = paragraph.text
        matches = [m for m in MATH_EXPRESSION.finditer(source) if not (re.fullmatch(r"\d+\.\d+", m.group()) and m.start() > 0 and m.end() < len(source) and source[m.start()-1:m.start()] == "(" and source[m.end():m.end()+1] == ")")]
        covered = [(m.start(), m.end()) for m in matches]
        matches += [m for m in INLINE_MATH.finditer(source) if not any(a <= m.start() and m.end() <= b for a, b in covered)]
        matches.sort(key=lambda m: m.start())
        if not matches:
            continue
        p = paragraph._p
        for child in list(p):
            if child.tag != qn("w:pPr"):
                p.remove(child)
        cursor = 0
        for match in matches:
            if match.start() > cursor:
                paragraph.add_run(source[cursor:match.start()])
            token = match.group()
            if MATH_EXPRESSION.fullmatch(token):
                latex = to_latex(token)
            else:
                latex = {
                "x_i^T": r"x_i^T", "w^T": r"w^T", "x_i": r"x_i", "y_i": r"y_i",
                "w_1": r"w_1", "w_2": r"w_2", "x_1": r"x_1", "x_2": r"x_2", "y_1": r"y_1", "y_2": r"y_2",
                "x_3": r"x_3", "w_i": r"w_i",
                "alpha_i": r"\alpha_i", "xi_i": r"\xi_i", "phi(x_i)": r"\phi(x_i)",
                "phi(x)": r"\phi(x)", "K(x_i, x_j)": r"K(x_i,x_j)", "gamma": r"\gamma",
                "L_hinge": r"L_{\mathrm{hinge}}", "||w||": r"\|w\|", "||x_i - x_j||": r"\|x_i-x_j\|",
                "||x_i||": r"\|x_i\|",
                "C": "C",
                ">=": r"\geq", "<=": r"\leq", "=>": r"\Rightarrow",
                ".": r"\cdot", "=": "=", "+": "+", "-": "-", "/": "/", "*": "\\times", "^": "^", "<": "<", ">": ">",
                }.get(token)
            if latex is None and re.fullmatch(r"[wxy]_(?:[a-z]|\d+)", token):
                base, sub = token.split("_", 1)
                latex = f"{base}_{{{sub}}}"
            if latex is None:
                raise ValueError(f"Chưa có quy tắc LaTeX cho ký hiệu: {token}")
            mathml = latex_to_mathml(latex)
            omml = mathml_to_omml(mathml).replace(
                "<m:oMath>", '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">', 1
            )
            p.append(parse_xml(omml.encode("utf-8")))
            cursor = match.end()
        if cursor < len(source):
            paragraph.add_run(source[cursor:])


DISPLAY_FORMULAS = [
    ("w^T . x - b = 0", r"w^T x-b=0", "(2.1)"),
    ("(1/2) . ||w||^2", r"\frac{1}{2}\|w\|^2", "(2.2)"),
    ("y_i . (w^T . x_i - b) >= 1, với mọi i = 1, 2, ..., N", r"y_i(w^Tx_i-b)\geq1,\quad i=1,\ldots,N", "(2.3)"),
    ("(1/2) . ||w||^2 + C . Tổng(xi_i)", r"\frac{1}{2}\|w\|^2+C\sum_i\xi_i", "(2.4)"),
    ("y_i . (w^T . x_i - b) >= 1 - xi_i và xi_i >= 0", r"y_i(w^Tx_i-b)\geq1-\xi_i,\quad\xi_i\geq0", "(2.5)"),
    ("L_hinge = max(0, 1 - y_i . (w^T . x_i - b))", r"L_{\mathrm{hinge}}=\max(0,1-y_i(w^Tx_i-b))", "(2.6)"),
    ("K(x_i, x_j) = phi(x_i)^T . phi(x_j)", r"K(x_i,x_j)=\phi(x_i)^T\phi(x_j)", "(3.1)"),
    ("K(x_i, x_j) = x_i^T . x_j", r"K(x_i,x_j)=x_i^Tx_j", "(3.2)"),
    ("K(x_i, x_j) = (gamma . x_i^T . x_j + r)^d", r"K(x_i,x_j)=(\gamma x_i^Tx_j+r)^d", "(3.3)"),
    ("K(x_i, x_j) = exp( -gamma . ||x_i - x_j||^2 )", r"K(x_i,x_j)=\exp(-\gamma\|x_i-x_j\|^2)", "(3.4)"),
    ("K * (K - 1) / 2", r"\frac{K(K-1)}{2}", "(3.5)"),
    ("+1 . (2w_1 + 2w_2 - b) >= 1", r"(+1)(2w_1+2w_2-b)\geq1", "(4.1)"),
    ("+1 . (2w_1 + 0w_2 - b) >= 1", r"(+1)(2w_1-b)\geq1", "(4.2)"),
    ("-1 . (0w_1 + 0w_2 - b) >= 1 => b >= 1", r"(-1)(-b)\geq1\Rightarrow b\geq1", "(4.3)"),
    ("0.5 . x_1 + 0.5 . x_2 - 1 = 0", r"0.5x_1+0.5x_2-1=0", "(4.4)"),
    ("2 / ||w|| = 2 / căn bậc hai(0.5^2 + 0.5^2) khoảng 2.828", r"\frac{2}{\|w\|}=\frac{2}{\sqrt{0.5^2+0.5^2}}\approx2.828", "(4.5)"),
]


def format_display_equations(document: Document) -> None:
    for paragraph in list(document.paragraphs):
        source = paragraph.text
        for raw, latex, number in sorted(DISPLAY_FORMULAS, key=lambda item: len(item[0]), reverse=True):
            if raw not in source:
                continue
            prefix = source[:source.index(raw)].strip()
            paragraph.text = prefix
            equation = document.add_paragraph()
            equation.paragraph_format.tab_stops.add_tab_stop(Cm(8), WD_TAB_ALIGNMENT.CENTER)
            equation.paragraph_format.tab_stops.add_tab_stop(Cm(16), WD_TAB_ALIGNMENT.RIGHT)
            equation.alignment = WD_ALIGN_PARAGRAPH.CENTER
            equation.add_run("\t")
            mathml = latex_to_mathml(latex)
            omml = mathml_to_omml(mathml).replace("<m:oMath>", '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">', 1)
            equation._p.append(parse_xml(omml.encode("utf-8")))
            equation.add_run(f"\t{number}")
            paragraph._p.addnext(equation._p)
            if not prefix:
                paragraph._p.getparent().remove(paragraph._p)
            break


def fmt(value, digits=4):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def configure_styles(document: Document) -> None:
    for section in document.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(3)
        section.right_margin = Cm(2)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.footer_distance = Cm(1.2)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(13)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(8)
    for name in ("Content", "Caption", "Heading 1", "Heading 2", "Heading 3", "Heading 1N", "CoverB"):
        if name in [style.name for style in document.styles]:
            style = document.styles[name]
            style.font.name = "Times New Roman"
            style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
            style.font.size = Pt(14 if name == "Heading 1" else 13)
    for name in ("Content",):
        style = document.styles[name]
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.space_after = Pt(8)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        style = document.styles[name]
        style.font.bold = True
        style.font.italic = name == "Heading 3"
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.space_before = Pt(10 if name == "Heading 1" else 6)
        style.paragraph_format.space_after = Pt(6)
    caption = document.styles["Caption"]
    caption.font.italic = True
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.line_spacing = 1.0
    caption.paragraph_format.space_after = Pt(8)
    footer_part = document.sections[0].footer
    for child in list(footer_part._element):
        footer_part._element.remove(child)
    footer = footer_part.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    page = OxmlElement("w:fldSimple")
    page.set(qn("w:instr"), "PAGE")
    footer._p.append(page)


def add_cover(document: Document) -> None:
    """Bìa trung tính theo mẫu; không suy đoán thông tin sinh viên hay giảng viên."""
    for paragraph in document.paragraphs:
        if paragraph._p.xpath('.//w:bookmarkStart[@w:name="svm_report_cover"]'):
            paragraph._p.getparent().remove(paragraph._p)
            break
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1.0
    bookmark = OxmlElement("w:bookmarkStart")
    bookmark.set(qn("w:id"), "32000")
    bookmark.set(qn("w:name"), "svm_report_cover")
    paragraph._p.insert(0, bookmark)
    first = paragraph.add_run("BÁO CÁO MÔN TỐI ƯU HÓA\n\n\n\n\n\n")
    first.bold, first.font.size = True, Pt(14)
    title = paragraph.add_run("NGHIÊN CỨU SVM DƯỚI GÓC NHÌN TỐI ƯU HÓA")
    title.bold, title.font.size = True, Pt(16)
    details = paragraph.add_run("\n\n\n\n\n\n\nSinh viên: ................................................\n\nGiảng viên hướng dẫn: ................................................")
    details.font.size = Pt(13)
    bookmark_end = OxmlElement("w:bookmarkEnd")
    bookmark_end.set(qn("w:id"), "32000")
    paragraph._p.append(bookmark_end)
    cover_section = deepcopy(document.sections[0]._sectPr)
    for child in list(cover_section):
        if child.tag in {qn("w:headerReference"), qn("w:footerReference"), qn("w:pgNumType")}:
            cover_section.remove(child)
    section_type = cover_section.find(qn("w:type"))
    if section_type is None:
        section_type = OxmlElement("w:type")
        cover_section.insert(0, section_type)
    section_type.set(qn("w:val"), "nextPage")
    paragraph._p.get_or_add_pPr().append(cover_section)
    document._element.body.insert(0, paragraph._p)


def find_paragraph(document: Document, text: str):
    """Tìm mốc chính xác; dừng thay vì chèn nhầm nếu bản nền đã đổi."""
    matches = [p for p in document.paragraphs if p.text.strip() == text]
    if len(matches) != 1:
        raise ValueError(f"Không tìm được duy nhất mục trong bản nền: {text}")
    return matches[0]


def insert_before(document: Document, anchor, build) -> None:
    """Chuyển các phần mới tạo đến trước mốc mà không xóa phần nền."""
    body = document._element.body
    existing = set(body)
    build()
    for element in list(body):
        if element not in existing and element.tag != qn("w:sectPr"):
            anchor._p.addprevious(element)


def merge_base_content(document: Document) -> None:
    """Restore the chapter body from its editable content source into the layout."""
    if not CONTENT.exists():
        raise FileNotFoundError(f"Thiếu nội dung báo cáo: {CONTENT}")
    anchor = find_paragraph(document, "KẾT LUẬN VÀ KIẾN NGHỊ")
    records = json.loads(CONTENT.read_text(encoding="utf-8"))
    for record in records:
        anchor._p.addprevious(parse_xml(record["xml"].encode("utf-8")))


def refresh_toc(document: Document) -> None:
    # Trường TOC trong file nhóm có cache từ một đề án khác; làm mới field,
    # giữ nguyên các chương và các đoạn nội dung của nhóm.
    for control in document._element.body.findall(qn("w:sdt")):
        if any("TOC" in (n.text or "") for n in control.iter(qn("w:instrText"))):
            content = control.find(qn("w:sdtContent"))
            for child in list(content):
                content.remove(child)
            placeholder = document.add_paragraph()
            placeholder.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_field(placeholder, r'TOC \o "1-3" \h \z \u')
            content.append(placeholder._p)
            break
    settings = document.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")

    # Danh mục trong bản nền còn là ví dụ từ template, với bookmark không tồn tại.
    paragraphs = document.paragraphs
    index = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "DANH MỤC CÁC HÌNH ẢNH")
    paragraphs[index + 1].text = "\n".join([
        "Hình 1.1. Lộ trình nghiên cứu SVM",
        "Hình 2.1. Siêu phẳng và support vectors",
        "Hình 2.2. Soft margin và slack",
        "Hình 2.3. Hinge loss",
        "Hình 3.1. Linear và RBF trên XOR",
        "Hình 4.1. Kiến trúc module",
        "Hình 4.2. Pipeline Banknote",
        "Hình 4.3. Một vòng SMO",
        "Hình 4.4. So sánh metrics",
        "Hình 4.5. Ma trận nhầm lẫn",
        "Hình 4.6. F1 trung bình 5-fold CV",
        "Hình 4.7. Hội tụ SMO",
        "Hình 4.8. Ranh giới quyết định 2D",
        "Hình 4.9. ROC trên tập test",
    ])
    paragraphs[index + 1].alignment = WD_ALIGN_PARAGRAPH.LEFT
    index = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "DANH MỤC BẢNG BIỂU")
    paragraphs[index + 1].text = "Bảng 4.1. Kết quả trên cùng tập test Banknote."
    paragraphs[index + 1].alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraphs[index + 2].text = ""


def main() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Không tìm thấy báo cáo nền: {TEMPLATE}")
    metrics_path = DATA / "model_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Chưa có benchmark. Chạy: uv run python -m experiments.06_report_benchmark")
    metrics = pd.read_csv(metrics_path)
    summary = json.loads((DATA / "dataset_summary.json").read_text(encoding="utf-8"))
    artifacts = json.loads((ROOT / "reports" / "benchmark_artifacts.json").read_text(encoding="utf-8"))
    document = Document(str(TEMPLATE))
    merge_base_content(document)
    configure_styles(document)
    refresh_toc(document)
    add_cover(document)
    document.core_properties.title = "Nghiên cứu SVM dưới góc nhìn tối ưu hóa"

    def chapter_one():
        document.add_paragraph("Lộ trình của project đi từ hình học SVM, bài toán primal và dual đến điều kiện KKT, thuật toán SMO và đánh giá trên Banknote Authentication.")
        add_figure(document, DIAGRAMS / "overview.png", "Hình 1.1. Lộ trình nghiên cứu SVM.", width=6.1)
    insert_before(document, find_paragraph(document, "Kết luận chương 1"), chapter_one)

    def geometry():
        add_figure(document, ILLUSTRATIONS / "svm_geometry.png", "Hình 2.1. Siêu phẳng, hai biên và support vectors trên dữ liệu tổng hợp.", width=5.2)
    insert_before(document, find_paragraph(document, "Máy vector hỗ trợ lề cứng (Hard Margin SVM)"), geometry)

    def soft_margin():
        document.add_paragraph("Trong code, hàm quyết định dùng quy ước wᵀx+b; quy ước này tương đương wᵀx−b trong bản nền khi đổi dấu bias. Hàm mục tiêu soft margin được viết:")
        add_figure(document, ILLUSTRATIONS / "soft_margin.png", "Hình 2.2. Một điểm vi phạm biên với slack dương; dữ liệu tổng hợp.", width=4.6)
        add_figure(document, ILLUSTRATIONS / "hinge_loss.png", "Hình 2.3. Hinge loss theo margin có dấu.", width=4.6)
    insert_before(document, find_paragraph(document, "Phương pháp nhân tử Lagrange và bài toán đối ngẫu"), soft_margin)

    def kernel():
        document.add_paragraph("Hình sau so sánh Linear và RBF trên cùng dữ liệu XOR tổng hợp; đây là hình minh họa trên train, không phải kết quả Banknote.")
        add_figure(document, ILLUSTRATIONS / "linear_rbf_xor.png", "Hình 3.1. Linear và RBF trên dữ liệu XOR tổng hợp.", width=6.0)
    insert_before(document, find_paragraph(document, "Mở rộng SVM cho bài toán phân lớp đa lớp và hồi quy (SVR)"), kernel)

    # Giữ nguyên ví dụ gốc, đồng thời làm rõ kết quả tính sai trong bản nền.
    def hand_correction():
        document.add_paragraph("Hiệu chỉnh ví dụ trên: với ba điểm đã cho và quy ước f(x)=wᵀx−b, nghiệm lề cứng là w=[1,0], b=1, lề rộng 2. Nghiệm w=[0.5,0.5], b=1 nêu trên cho f([2,0])=0 nên không thỏa ràng buộc y·f(x)≥1. Ví dụ bốn điểm khác của project được trình bày trong docs/07_hand_example.md.")
    insert_before(document, find_paragraph(document, "Thiết lập thực nghiệm và tiền xử lý dữ liệu"), hand_correction)

    def setup():
        document.add_paragraph(f"Thực nghiệm bổ sung dùng UCI Banknote Authentication: {summary['n_samples']} mẫu, {summary['n_features']} đặc trưng, train/test {summary['train_samples']}/{summary['test_samples']}, chia phân tầng với random_state={summary['random_state']}. StandardScaler chỉ fit trên train; trong cross-validation, scaler được fit lại trên train của từng fold.")
        document.add_paragraph("SMO cập nhật hai hệ số alpha mỗi lần để giữ ràng buộc của bài toán dual:")
        add_equation(document, r"\sum_i\alpha_i y_i=0,\qquad 0\leq\alpha_i\leq C", "(4.6)")
        add_figure(document, DIAGRAMS / "architecture.png", "Hình 4.1. Kiến trúc module của project.", width=6.0)
        add_figure(document, DIAGRAMS / "experiment_pipeline.png", "Hình 4.2. Pipeline thực nghiệm Banknote.", width=6.0)
        add_figure(document, DIAGRAMS / "smo_pair_update.png", "Hình 4.3. Một vòng cập nhật cặp SMO.", width=6.0)
    insert_before(document, find_paragraph(document, "Kết quả thực nghiệm và so sánh đánh giá"), setup)

    def results():
        best = artifacts["best_rbf_params"]
        document.add_paragraph(f"Kết quả bổ sung của project: RBF chọn C={best.get('svc__C')} và gamma={best.get('svc__gamma')} bằng F1 trung bình qua 5-fold trên train. Tập test giữ riêng để đánh giá sau khi chọn tham số.")
        rows = [(r["model"], fmt(r["accuracy"]), fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]), fmt(r["roc_auc"])) for _, r in metrics.iterrows()]
        add_caption(document, "Bảng 4.1. Kết quả trên cùng tập test Banknote.")
        add_table(document, ["Mô hình", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], rows, widths=[1.7, .85, .85, .85, .85, .85])
        add_figure(document, FIGURES / "benchmark_metrics.png", "Hình 4.4. So sánh metrics trên tập test.", width=6.0)
        add_figure(document, FIGURES / "benchmark_confusion_matrices.png", "Hình 4.5. Ma trận nhầm lẫn trên tập test.", width=6.0)
        add_figure(document, FIGURES / "rbf_grid_search_heatmap.png", "Hình 4.6. F1 trung bình 5-fold CV của RBF SVC.", width=5.0)
        add_figure(document, FIGURES / "custom_smo_convergence.png", "Hình 4.7. Lịch sử hàm mục tiêu dual của SMO.", width=5.8)
        add_figure(document, FIGURES / "decision_boundaries_2d.png", "Hình 4.8. Ranh giới trên hai đặc trưng đầu, chỉ để minh họa.", width=6.0)
        add_figure(document, FIGURES / "benchmark_roc.png", "Hình 4.9. ROC trên tập test.", width=5.8)
    insert_before(document, find_paragraph(document, "Kết luận Chương 4"), results)

    def references():
        for text in [
            "Cortes, C. & Vapnik, V. (1995). Support-Vector Networks. Machine Learning, 20, 273–297.",
            "Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. MSR-TR-98-14.",
            "Chang, C.-C. & Lin, C.-J. (2011). LIBSVM: A Library for Support Vector Machines.",
            "UCI Banknote Authentication: https://archive.ics.uci.edu/dataset/267/banknote+authentication",
        ]:
            document.add_paragraph(text)
    insert_before(document, find_paragraph(document, "PHỤ LỤC"), references)

    document.add_paragraph("Tái lập kết quả: chạy uv sync --locked, uv run python -m experiments.06_report_benchmark và uv run python reports/generate_docx.py từ thư mục gốc project.")
    format_display_equations(document)
    format_inline_math(document)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(OUTPUT))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
