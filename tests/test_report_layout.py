import importlib.util
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.shared import Cm, Pt

spec = importlib.util.spec_from_file_location("generate_docx", Path(__file__).resolve().parents[1] / "reports" / "generate_docx.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_reference_page_layout_and_cover_are_repeatable():
    document = Document()
    document.styles.add_style("Content", WD_STYLE_TYPE.PARAGRAPH)
    report.configure_styles(document)
    section = document.sections[0]
    assert abs(section.page_width - Cm(21)) < 1000
    assert abs(section.left_margin - Cm(3)) < 1000
    assert document.styles["Normal"].font.size == Pt(13)
    assert document.styles["Normal"].paragraph_format.line_spacing == 1.5
    report.add_cover(document)
    report.add_cover(document)
    assert len(document._element.xpath('.//w:bookmarkStart[@w:name="svm_report_cover"]')) == 1
    assert len(document.sections) == 2


def test_insert_before_preserves_existing_content_and_order():
    document = Document()
    document.add_paragraph("Nội dung của nhóm")
    anchor = document.add_paragraph("Kết luận chương")

    def append_new():
        document.add_paragraph("Hình bổ sung")
        document.add_paragraph("Giải thích bổ sung")

    report.insert_before(document, anchor, append_new)
    assert [p.text for p in document.paragraphs] == [
        "Nội dung của nhóm", "Hình bổ sung", "Giải thích bổ sung", "Kết luận chương"
    ]
