import importlib.util
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.shared import Cm, Pt

spec = importlib.util.spec_from_file_location("generate_docx", Path(__file__).resolve().parents[1] / "reports" / "scripts" / "generate_docx.py")
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


def test_markdown_additions_number_captions_render_cells_and_keep_code(tmp_path):
    import check_math
    from docx.oxml.ns import qn
    from report_math import render_math

    document = Document()
    document.styles.add_style("Content", WD_STYLE_TYPE.PARAGRAPH)
    report.configure_styles(document)
    for title in ("DANH MỤC CÁC HÌNH ẢNH", "DANH MỤC BẢNG BIỂU"):
        document.add_paragraph(title)
        document.add_paragraph("mục cũ của template")
    document.add_paragraph("")
    document.add_heading("Cơ sở lý thuyết", 1)
    figure = report.ILLUSTRATIONS / "hinge_loss.png"
    report.add_markdown(document, f"""Đoạn có **ý chính** và [một liên kết](https://example.com).

*Bảng: Ba trường hợp*
| $t$ | an toàn |
|---|---:|
| $1$ | $2$ |

![Hinge]({figure}){{width=2}}

```python
loss = max(0, 1 - y * f_x)
```
""", tmp_path)
    report.number_captions(document)
    render_math(document)

    body = document.element.body
    prose = next(p for p in document.paragraphs if p.text.startswith("Đoạn có"))
    assert prose.text == "Đoạn có ý chính và một liên kết."
    assert [run.text for run in prose.runs if run.bold] == ["ý chính"]
    captions = [p.text for p in document.paragraphs if p.style.name == "Caption"]
    assert captions == ["Bảng 1.1. Ba trường hợp", "Hình 1.1. " + report.figure_caption(figure)]
    assert all(cell._tc.findall(".//" + check_math.M + "oMath") for row in document.tables[0].rows for cell in row.cells[:1])
    code = next(p for p in document.paragraphs if p.style.name == "Code")
    assert code.text == "loss = max(0, 1 - y * f_x)" and not code._p.findall(".//" + check_math.M + "oMath")
    errors, found = check_math.check_captions(body)
    assert errors == [] and len(found["Hình"]) == len(found["Bảng"]) == 1
    assert check_math.check_leftovers(body.iter(qn("w:p"))) == []

    label = next(p for p in document.paragraphs if p.style.name == "Caption" and p.text.startswith("Hình 1.1."))
    label.runs[0].text = label.runs[0].text.replace("Hình 1.1.", "Hình 1.2.")
    assert any("Hình 1.1" in error for error in check_math.check_captions(body)[0])
