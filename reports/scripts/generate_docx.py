r"""Sinh báo cáo Word theo file định dạng của cô Xuân.

Chạy sau khi đã chạy benchmark:

    uv run python reports/scripts/generate_docx.py

Dùng source/report_layout.docx và source/report_content.json làm nội dung nền, giữ nguyên các chương của nhóm.
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
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from report_math import CODE_STYLE, chapter_heading, render_math

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "source"
OUTPUT = ROOT / "reports" / "output" / "svm_optimization_report.docx"
TEMPLATE = Path(os.environ.get("REPORT_TEMPLATE", SOURCE / "report_layout.docx"))
CONTENT = SOURCE / "report_content.json"
ADDITIONS = SOURCE / "report_additions.md"
DATA = ROOT / "reports" / "data"
FIGURES = ROOT / "reports" / "figures"
ILLUSTRATIONS = ROOT / "docs" / "images" / "scientific"
DIAGRAMS = ROOT / "docs" / "diagrams"
CODE_FONT = "Consolas"


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


def add_caption(document: Document, text: str):
    p = document.add_paragraph(style="Caption" if "Caption" in [s.name for s in document.styles] else "Content")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(text)
    return p


FIGURE_CAPTIONS = {row.split("|", 2)[1].strip(" |\t"): row.split("|", 2)[2].strip(" |\t") for row in (SOURCE / "figure_captions.md").read_text(encoding="utf-8").splitlines() if row.startswith("|") and not row.startswith("| File")}


def figure_caption(path: Path) -> str:
    """Chú thích lấy từ dòng *Hình N: …* trong docs; hình không có trong docs lấy từ figure_captions.md."""
    for markdown in sorted((ROOT / "docs").glob("*.md")):
        lines = markdown.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines[:-1]):
            if re.search(r"!\[[^]]*\]\([^)]*/" + re.escape(path.stem) + r"\.(?:svg|png)\)", line):
                caption_line = next((value.strip().strip("*") for value in lines[index + 1:index + 5] if value.strip().startswith("*Hình ")), "")
                if caption_line:
                    return caption_line.split(":", 1)[-1].strip()
    configured = next((value for name, value in FIGURE_CAPTIONS.items() if Path(name).stem == path.stem), None)
    if configured is None:
        raise ValueError(f"Thiếu chú thích cho hình {path.name} trong docs hoặc figure_captions.md")
    return configured


def add_figure(document: Document, path: Path, width: float = 6.1) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Thiếu hình: {path}")
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(document, figure_caption(path))


def add_code(document: Document, code: str) -> None:
    p = document.add_paragraph(style=CODE_STYLE)
    run = p.add_run(code)  # "\n" thành ngắt dòng trong cùng một khối
    run.font.name = CODE_FONT
    run._element.rPr.rFonts.set(qn("w:hAnsi"), CODE_FONT)


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
    if CODE_STYLE not in [style.name for style in document.styles]:
        document.styles.add_style(CODE_STYLE, WD_STYLE_TYPE.PARAGRAPH)
    code = document.styles[CODE_STYLE]
    code.base_style = document.styles["Normal"]
    code.font.name = CODE_FONT
    code.font.size = Pt(10)
    code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    code.paragraph_format.line_spacing = 1.0
    code.paragraph_format.first_line_indent = Cm(0)
    code.paragraph_format.left_indent = Cm(0.5)
    code.paragraph_format.space_after = Pt(8)
    code.paragraph_format.keep_together = True
    shading = code._element.get_or_add_pPr().find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        code._element.get_or_add_pPr().append(shading)
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:fill"), "F2F2F2")
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
    """Chuyển các phần mới tạo đến trước phần tử mốc mà không xóa phần nền."""
    body = document._element.body
    existing = set(body)
    build()
    for element in list(body):
        if element not in existing and element.tag != qn("w:sectPr"):
            anchor.addprevious(element)


def merge_base_content(document: Document) -> None:
    """Restore the chapter body from its editable content source into the layout."""
    if not CONTENT.exists():
        raise FileNotFoundError(f"Thiếu nội dung báo cáo: {CONTENT}")
    anchor = find_paragraph(document, "KẾT LUẬN VÀ KIẾN NGHỊ")
    records = json.loads(CONTENT.read_text(encoding="utf-8"))
    for record in records:
        anchor._p.addprevious(parse_xml(record["xml"].encode("utf-8")))


IMAGE = re.compile(r"^!\[[^]]*\]\(([^)]+)\)(?:\{width=([\d.]+)\})?$")
TABLE_CAPTION = re.compile(r"^\*Bảng:\s*(.+)\*$")


def add_text(document: Document, text: str) -> None:
    """Đoạn văn Markdown: **đậm**, [liên kết](url) giữ chữ; toán và `code` để render_math xử lý."""
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    p = document.add_paragraph()
    for index, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            p.add_run(part).bold = index % 2 == 1


def add_markdown(document: Document, text: str, base: Path) -> None:
    """Chèn các khối Markdown: đoạn văn, $$công thức$$, hình, bảng (có *Bảng: …* phía trên) và khối code."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if lines[i].startswith("```"):
            end = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("```"))
            add_code(document, "\n".join(lines[i + 1:end]))
            i = end + 1
            continue
        end = i
        while end < len(lines) and lines[end].strip() and not lines[end].startswith("```"):
            end += 1
        block = [line.strip() for line in lines[i:end]]
        i = end
        if TABLE_CAPTION.match(block[0]):
            add_caption(document, TABLE_CAPTION.match(block[0]).group(1)).paragraph_format.keep_with_next = True
            block = block[1:]
        image = IMAGE.match(block[0]) if len(block) == 1 else None
        if image:
            add_figure(document, (base / image.group(1)).resolve(), width=float(image.group(2) or 6.0))
        elif block and block[0].startswith("|"):
            # "\|" trong ô là chuẩn LaTeX, không phải ranh giới cột.
            rows = [[cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip("|"))] for line in block]
            add_table(document, rows[0], rows[2:])
        elif block:
            add_text(document, " ".join(block))


def insert_additions(document: Document) -> None:
    """Mỗi mục `## Trước: <tiêu đề>` / `## Sau: <tiêu đề>` của report_additions.md được chèn ngay trước / sau tiêu đề đó."""
    sections = re.split(r"^## (Trước|Sau): (.+)$", ADDITIONS.read_text(encoding="utf-8"), flags=re.MULTILINE)
    for where, heading, text in zip(sections[1::3], sections[2::3], sections[3::3]):
        anchor = find_paragraph(document, heading.strip())._p
        insert_before(document, anchor if where == "Trước" else anchor.getnext(), lambda text=text: add_markdown(document, text, ADDITIONS.parent))


def number_captions(document: Document) -> None:
    """Đánh số Hình/Bảng theo chương (c.n) theo thứ tự xuất hiện, rồi dựng lại hai danh mục."""
    chapter, counts, found = None, {}, {"Hình": [], "Bảng": []}
    for paragraph in document.paragraphs:
        heading = chapter_heading(paragraph)
        if heading is not None:
            chapter, counts = ((chapter or 0) + 1 if heading else None), {}
        if paragraph.style.name != "Caption":
            continue
        previous, following = paragraph._p.getprevious(), paragraph._p.getnext()
        if previous is not None and previous.findall(".//" + qn("w:drawing")):
            kind = "Hình"
        elif following is not None and following.tag == qn("w:tbl"):
            kind = "Bảng"
        else:
            raise ValueError(f"Chú thích không đứng sau hình hay trước bảng: {paragraph.text!r}")
        if chapter is None:
            raise ValueError(f"Chú thích nằm ngoài chương được đánh số: {paragraph.text!r}")
        counts[kind] = counts.get(kind, 0) + 1
        paragraph.runs[0].text = f"{kind} {chapter}.{counts[kind]}. " + paragraph.runs[0].text
        found[kind].append(paragraph.text)

    # Danh mục trong bản nền còn là ví dụ từ template, với bookmark không tồn tại.
    paragraphs = document.paragraphs
    index = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "DANH MỤC CÁC HÌNH ẢNH")
    paragraphs[index + 1].text = "\n".join(found["Hình"])
    paragraphs[index + 1].alignment = WD_ALIGN_PARAGRAPH.LEFT
    index = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "DANH MỤC BẢNG BIỂU")
    paragraphs[index + 1].text = "\n".join(found["Bảng"])
    paragraphs[index + 1].alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraphs[index + 2].text = ""

    # Gán .text làm mất begin/separate của field danh mục cũ nhưng end nằm ở đoạn sau vẫn còn.
    depth = 0
    for char in list(document._element.body.iter(qn("w:fldChar"))):
        kind = char.get(qn("w:fldCharType"))
        depth += {"begin": 1, "end": -1}.get(kind, 0)
        if depth < 0:
            char.getparent().remove(char)
            depth = 0


def left_align_back_matter(document: Document) -> None:
    """URL và lệnh dài không ngắt được: căn đều sẽ giãn chữ, nên phần sau TÀI LIỆU THAM KHẢO căn trái."""
    paragraphs = document.paragraphs
    start = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "TÀI LIỆU THAM KHẢO")
    for paragraph in paragraphs[start + 1:]:
        if not paragraph.style.name.startswith("Heading"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT


CITATION = re.compile(r"\[(\d+)\]")


def link_citations(document: Document) -> None:
    """Đoạn "[n] …" dưới TÀI LIỆU THAM KHẢO nhận bookmark ref_n; mỗi "[n]" trong văn bản phía trên trỏ tới đó."""
    paragraphs = document.paragraphs
    start = next(i for i, p in enumerate(paragraphs) if p.text.strip() == "TÀI LIỆU THAM KHẢO")
    refs = set()
    for paragraph in paragraphs[start + 1:]:
        if paragraph.style.name.startswith("Heading"):
            break
        match = CITATION.match(paragraph.text)
        if match:
            number = match.group(1)
            start_mark, end_mark = OxmlElement("w:bookmarkStart"), OxmlElement("w:bookmarkEnd")
            for mark in (start_mark, end_mark):
                mark.set(qn("w:id"), str(32001 + int(number)))
            start_mark.set(qn("w:name"), f"ref_{number}")
            paragraph._p.get_or_add_pPr().addnext(start_mark)
            paragraph._p.append(end_mark)
            refs.add(number)
    for paragraph in paragraphs[:start]:
        if paragraph.style.name == CODE_STYLE:
            continue
        for run in paragraph._p.findall(qn("w:r")):
            texts = run.findall(qn("w:t"))
            fonts = run.find(f"{qn('w:rPr')}/{qn('w:rFonts')}")
            if len(texts) != 1 or (fonts is not None and fonts.get(qn("w:ascii")) == CODE_FONT):
                continue
            parts = CITATION.split(texts[0].text or "")
            if not any(number in refs for number in parts[1::2]):
                continue
            for index, part in enumerate(parts):
                if not part:
                    continue
                piece = deepcopy(run)
                piece.find(qn("w:t")).text = f"[{part}]" if index % 2 else part
                piece.find(qn("w:t")).set(qn("xml:space"), "preserve")
                if index % 2 and part in refs:
                    link = OxmlElement("w:hyperlink")
                    link.set(qn("w:anchor"), f"ref_{part}")
                    link.append(piece)
                    piece = link
                run.addprevious(piece)
            run.getparent().remove(run)


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


def main() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Không tìm thấy báo cáo nền: {TEMPLATE}")
    metrics_path = DATA / "model_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Chưa có benchmark. Chạy: uv run python -m experiments.06_report_benchmark")
    metrics = pd.read_csv(metrics_path)
    summary = json.loads((DATA / "dataset_summary.json").read_text(encoding="utf-8"))
    artifacts = json.loads((DATA / "benchmark_artifacts.json").read_text(encoding="utf-8"))
    document = Document(str(TEMPLATE))
    merge_base_content(document)
    configure_styles(document)
    refresh_toc(document)
    add_cover(document)
    document.core_properties.title = "Nghiên cứu SVM dưới góc nhìn tối ưu hóa"

    # Nội dung phụ thuộc dữ liệu benchmark; nội dung tĩnh nằm trong source/report_additions.md.
    def setup():
        document.add_paragraph(f"Thực nghiệm bổ sung dùng UCI Banknote Authentication: {summary['n_samples']} mẫu, {summary['n_features']} đặc trưng; tập train có {summary['train_samples']} mẫu và tập test có {summary['test_samples']} mẫu, chia phân tầng với `random_state={summary['random_state']}`. StandardScaler chỉ fit trên train; trong cross-validation, scaler được fit lại trên train của từng fold.")
    insert_before(document, find_paragraph(document, "Kết quả thực nghiệm và so sánh đánh giá")._p, setup)

    def results():
        best = artifacts["best_rbf_params"]
        document.add_paragraph(f"Kết quả bổ sung của project: RBF chọn $C = {best.get('svc__C')}$ và $\\gamma = {best.get('svc__gamma')}$ bằng F1 trung bình qua 5-fold trên train. Tập test giữ riêng để đánh giá sau khi chọn tham số.")
        rows = [(r["model"], fmt(r["accuracy"]), fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]), fmt(r["roc_auc"])) for _, r in metrics.iterrows()]
        add_caption(document, FIGURE_CAPTIONS["results_table"]).paragraph_format.keep_with_next = True
        add_table(document, ["Mô hình", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], rows, widths=[1.7, .85, .85, .85, .85, .85])
        add_figure(document, FIGURES / "benchmark_metrics.png", width=6.0)
        add_figure(document, FIGURES / "benchmark_confusion_matrices.png", width=6.0)
        add_figure(document, FIGURES / "rbf_grid_search_heatmap.png", width=5.0)
        add_figure(document, FIGURES / "custom_smo_convergence.png", width=5.8)
        add_figure(document, FIGURES / "decision_boundaries_2d.png", width=6.0)
        add_figure(document, FIGURES / "benchmark_roc.png", width=5.8)
    insert_before(document, find_paragraph(document, "Kết luận Chương 4")._p, results)

    insert_additions(document)
    number_captions(document)
    document.add_paragraph("Tái lập kết quả: chạy `uv sync --locked`, `uv run python -m experiments.06_report_benchmark` và `uv run python reports/scripts/generate_docx.py` từ thư mục gốc project.")
    render_math(document)
    link_citations(document)
    left_align_back_matter(document)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(OUTPUT))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
