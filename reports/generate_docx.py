r"""Sinh báo cáo Word theo file định dạng của cô Xuân.

Chạy sau khi đã chạy benchmark:

    python reports/generate_docx.py

Dùng báo cáo hiện có làm mẫu để giữ trang đầu và định dạng.
Có thể cung cấp template khác bằng biến môi trường ``REPORT_TEMPLATE``.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from latex2mathml.converter import convert as latex_to_mathml
from mathml2omml import convert as mathml_to_omml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "svm_optimization_report.docx"
TEMPLATE = Path(os.environ.get("REPORT_TEMPLATE", OUTPUT))
DATA = ROOT / "reports" / "data"
FIGURES = ROOT / "reports" / "figures"
ILLUSTRATIONS = ROOT / "docs" / "images" / "scientific"
DIAGRAMS = ROOT / "docs" / "diagrams"


def clear_body(document: Document) -> None:
    body = document._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def trim_after_template_front_matter(document: Document) -> None:
    """Giữ nguyên front matter và section break của template gốc.

    Template đặt section break ở paragraph ngay trước chương đầu tiên. Chỉ
    xóa phần nội dung mẫu sau section break, thay vì dựng lại document từ đầu.
    """
    chapter = next(
        paragraph for paragraph in document.paragraphs
        if paragraph.text.strip() in {
            "TỔNG QUAN VỀ HỌC MÁY VÀ BÀI TOÁN PHÂN LỚP",
            "CHƯƠNG 1. TỔNG QUAN VỀ HỌC MÁY VÀ BÀI TOÁN PHÂN LỚP",
        }
    )
    body = document._element.body
    remove = False
    for child in list(body):
        if child is chapter._p:
            remove = True
        if remove and child.tag != qn("w:sectPr"):
            body.remove(child)


def update_template_front_matter(document: Document) -> None:
    """Đổi chữ mẫu ở các trang đầu nhưng giữ style/layout của template."""
    paragraphs = document.paragraphs
    replacements = {
        1: "Tôi cam đoan báo cáo này sử dụng mã nguồn, số liệu và kết quả thực nghiệm được ghi rõ trong project SVM Optimization Lab. Các hình khái niệm là minh họa tự tạo; kết quả thực nghiệm được lấy từ các script benchmark.",
        5: "Tôi trân trọng cảm ơn giảng viên học phần đã cung cấp khung trình bày và định hướng để gắn lý thuyết tối ưu với một bài toán học máy có thể kiểm chứng.",
        10: (
            "Hình 1.1. Lộ trình học SVM\n"
            "Hình 2.1. Siêu phẳng và margin\n"
            "Hình 2.2. Soft margin và slack\n"
            "Hình 2.3. Hinge loss\n"
            "Hình 3.1. Linear và RBF trên dữ liệu phi tuyến\n"
            "Hình 3.2. Một vòng cập nhật SMO\n"
            "Hình 4.1. Kiến trúc mã nguồn\n"
            "Hình 4.2. Pipeline thực nghiệm Banknote\n"
            "Hình 5.1–5.6. Kết quả benchmark và trực quan hóa"
        ),
        14: "Bảng 4.1. Cấu hình benchmark và giao thức đánh giá",
        15: "Bảng 5.1. Metrics của các mô hình trên tập test",
        20: "SVM — Support Vector Machine",
        21: "QP — Quadratic Programming",
        22: "KKT — Karush–Kuhn–Tucker",
        23: "SMO — Sequential Minimal Optimization",
    }
    for index, text in replacements.items():
        if index < len(paragraphs):
            paragraphs[index].text = text
    if len(paragraphs) > 18:
        paragraphs[16].text = "DANH MỤC CÔNG THỨC"
        paragraphs[16].style = "Heading 1N" if "Heading 1N" in [s.name for s in document.styles] else paragraphs[16].style
        paragraphs[17].text = (
            "(2.1) Primal Soft-Margin — Chương 2\n"
            "(2.2) Dual SVM — Chương 2\n"
            "(3.1) Kernel Linear/RBF — Chương 3\n"
            "(4.1) Primal objective của ví dụ giải tay — Chương 4"
        )
        paragraphs[17].style = "Content" if "Content" in [s.name for s in document.styles] else paragraphs[17].style

    # The original template carries a cached TOC for a different report.
    # Replace it with a fresh field so Word can build entries for this report.
    body = document._element.body
    for child in list(body):
        if child.tag == qn("w:sdt") and any(
            "TOC" in (node.text or "") for node in child.iter(qn("w:instrText"))
        ):
            content = child.find(qn("w:sdtContent"))
            for item in list(content):
                content.remove(item)
            placeholder = document.add_paragraph()
            add_field(placeholder, r'TOC \o "1-3" \h \z \u')
            content.append(placeholder._p)



def add_heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_heading(text, level=level)
    # The template's Heading styles already contain numbering; our titles do too.
    num_pr = paragraph._p.get_or_add_pPr().get_or_add_numPr()
    num_pr.get_or_add_numId().val = 0


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(str(text))
    run.bold = bold
    p.paragraph_format.space_after = Pt(0)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    run._r.extend([begin, instr, separate])
    for index, entry in enumerate([
        "CHƯƠNG 1. TỔNG QUAN VỀ HỌC MÁY VÀ BÀI TOÁN PHÂN LỚP",
        "CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ HÌNH HỌC CỦA SVM",
        "CHƯƠNG 3. PHƯƠNG PHÁP HẠT NHÂN VÀ THUẬT TOÁN SMO",
        "CHƯƠNG 4. CÀI ĐẶT VÀ THIẾT KẾ THỰC NGHIỆM",
        "CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM",
        "CHƯƠNG 6. THẢO LUẬN VÀ HẠN CHẾ",
        "KẾT LUẬN VÀ KIẾN NGHỊ",
        "TÀI LIỆU THAM KHẢO",
        "PHỤ LỤC. CÁCH TÁI LẬP KẾT QUẢ",
        "Trong Word: chọn mục lục, nhấn F9 để cập nhật mục và số trang.",
    ]):
        if index:
            run._r.append(OxmlElement("w:br"))
        text = OxmlElement("w:t")
        text.text = entry
        run._r.append(text)
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(end)


def add_toc(document: Document) -> None:
    p = document.add_paragraph()
    add_field(p, 'TOC \\o "1-3" \\h \\z \\u')
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    note = document.add_paragraph("Trong Microsoft Word, chọn toàn bộ tài liệu rồi nhấn F9 để cập nhật số trang mục lục.")
    note.style = "Content"


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


def add_bullets(document: Document, items) -> None:
    for item in items:
        p = document.add_paragraph(style="Content")
        p.paragraph_format.left_indent = Inches(0.25)
        p.add_run("• " + item)


def add_equation(document: Document, latex: str, number: str) -> None:
    """Chèn công thức LaTeX thành Word OMML, có số công thức bên phải."""
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mathml = latex_to_mathml(latex)
    omml = mathml_to_omml(mathml).replace(
        "<m:oMath>",
        '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">',
        1,
    )
    p._p.append(parse_xml(omml.encode("utf-8")))
    p.add_run(f"    {number}").font.name = "Cambria Math"


def fmt(value, digits=4):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def configure_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(13)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        if name in [style.name for style in document.styles]:
            style = document.styles[name]
            style.font.name = "Times New Roman"
            style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def main() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Không tìm thấy template: {TEMPLATE}")
    metrics_path = DATA / "model_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Chưa có benchmark. Chạy: python -m experiments.06_report_benchmark")
    metrics = pd.read_csv(metrics_path)
    summary = json.loads((DATA / "dataset_summary.json").read_text(encoding="utf-8"))
    artifacts = json.loads((ROOT / "reports" / "benchmark_artifacts.json").read_text(encoding="utf-8"))

    document = Document(str(TEMPLATE))
    trim_after_template_front_matter(document)
    update_template_front_matter(document)
    document.core_properties.title = "Nghiên cứu SVM dưới góc nhìn tối ưu hóa"
    document.core_properties.subject = "Primal, Dual, KKT, SMO và benchmark Banknote Authentication"
    document.core_properties.author = "Nhóm thực hiện"

    add_heading(document, "CHƯƠNG 1. TỔNG QUAN VỀ HỌC MÁY VÀ BÀI TOÁN PHÂN LỚP", level=1)
    add_heading(document, "1.1. Học máy có giám sát và phân lớp", level=2)
    document.add_paragraph("Học máy có giám sát xây dựng ánh xạ từ vector đặc trưng x sang nhãn y dựa trên các mẫu đã biết. Trong phân lớp nhị phân, mục tiêu là tìm một hàm quyết định có thể phân biệt hai lớp và giữ khả năng tổng quát hóa trên mẫu chưa thấy.")
    add_heading(document, "1.2. Động cơ chọn SVM", level=2)
    document.add_paragraph("SVM phù hợp với học phần Tối ưu hóa vì nghiệm được xác định thông qua một bài toán cực tiểu lồi: tối thiểu hóa độ lớn của vector pháp tuyến đồng thời đáp ứng các ràng buộc phân lớp. Margin không chỉ là trực giác hình học mà còn là đại lượng nối trực tiếp giữa độ phức tạp mô hình và khả năng tổng quát hóa.")
    add_heading(document, "1.3. Phạm vi và lộ trình", level=2)
    document.add_paragraph("Báo cáo đi từ hình học, Primal, Lagrangian, Dual và KKT đến SMO; sau đó triển khai KernelSVM nhị phân và đối chiếu với các mô hình chuẩn trên UCI Banknote Authentication.")
    add_figure(document, DIAGRAMS / "overview.png", "Hình 1.1. Lộ trình học SVM từ hình học đến thực nghiệm.", width=6.1)

    add_heading(document, "CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ HÌNH HỌC CỦA SVM", level=1)
    add_heading(document, "2.1. Siêu phẳng, khoảng cách và margin", level=2)
    document.add_paragraph("Với w là vector pháp tuyến và b là bias, siêu phẳng quyết định có dạng f(x) = wᵀx + b = 0. Khoảng cách từ một điểm đến siêu phẳng tỉ lệ với |wᵀx+b|/||w||. Khi chuẩn hóa hai biên thành y_i f(x_i) ≥ 1, bề rộng margin là 2/||w||.")
    add_figure(document, ILLUSTRATIONS / "svm_geometry.png", "Hình 2.1. Siêu phẳng, margin và support vectors trên dữ liệu minh họa.", width=5.5)
    add_heading(document, "2.2. Hard margin và Soft margin", level=2)
    document.add_paragraph("Hard-margin SVM giả định dữ liệu phân tách tuyến tính hoàn toàn. Với dữ liệu nhiễu, biến slack ξ_i cho phép một số ràng buộc bị vi phạm và tham số C điều chỉnh mức phạt cho vi phạm đó.")
    add_equation(document, r"\min_{w,b,\xi}\;\frac{1}{2}\|w\|^2+C\sum_{i=1}^{n}\xi_i", "(2.1)")
    add_figure(document, ILLUSTRATIONS / "soft_margin.png", "Hình 2.2. Soft margin và slack trên dữ liệu minh họa.", width=5.5)
    add_figure(document, ILLUSTRATIONS / "hinge_loss.png", "Hình 2.3. Hinge loss theo margin y f(x).", width=5.5)
    document.add_paragraph("C lớn ưu tiên giảm lỗi huấn luyện; C nhỏ ưu tiên margin rộng hơn. Dạng không ràng buộc tương đương liên hệ trực tiếp với hinge loss max(0, 1−y_i f(x_i)).")
    add_heading(document, "2.3. Lagrangian, Dual và KKT", level=2)
    document.add_paragraph("Đưa các ràng buộc vào Lagrangian với các multiplier α_i và μ_i, sau đó đặt các đạo hàm riêng theo w, b và ξ bằng 0. Kết quả là bài toán Dual chỉ còn phụ thuộc vào tích vô hướng x_iᵀx_j, mở đường cho kernel.")
    add_equation(document, r"\max_{\alpha}\;\sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j)", "(2.2)")
    add_bullets(document, [
        "Primal feasibility: các ràng buộc của bài toán gốc được thỏa mãn.",
        "Dual feasibility: α_i nằm trong [0,C] và thỏa mãn ràng buộc đẳng thức.",
        "Stationarity: gradient của Lagrangian theo biến quyết định bằng 0.",
        "Complementary slackness: α_i chỉ dương khi điểm nằm trên biên hoặc vi phạm margin.",
    ])

    add_heading(document, "CHƯƠNG 3. PHƯƠNG PHÁP HẠT NHÂN VÀ THUẬT TOÁN SMO", level=1)
    add_heading(document, "3.1. Kernel Trick và điều kiện Mercer", level=2)
    document.add_paragraph("Thay vì biểu diễn tường minh ánh xạ φ(x), kernel tính trực tiếp K(x_i,x_j)=φ(x_i)ᵀφ(x_j). Một kernel hợp lệ cần tạo ra ma trận Gram nửa xác định dương; đây là điều kiện cần để bài toán Dual vẫn giữ tính lồi.")

    add_heading(document, "3.2. Linear và RBF kernel", level=2)
    add_equation(document, r"K_{\mathrm{linear}}(x,z)=x^Tz\;\quad K_{\mathrm{RBF}}(x,z)=\exp(-\gamma\|x-z\|^2)", "(3.1)")
    document.add_paragraph("γ kiểm soát bán kính ảnh hưởng của từng mẫu. γ lớn tạo biên linh hoạt và có nguy cơ quá khớp; γ nhỏ tạo biên mượt hơn nhưng có thể dưới khớp.")
    add_figure(document, ILLUSTRATIONS / "linear_rbf_xor.png", "Hình 3.1. So sánh Linear và RBF trên dữ liệu phi tuyến tổng hợp.", width=6.0)
    add_heading(document, "3.3. Sequential Minimal Optimization", level=2)
    document.add_paragraph("SMO chọn hai alpha mỗi lần cập nhật để bảo toàn Σ_i α_i y_i=0. Mỗi vòng gồm kiểm tra KKT, tính khoảng [L,H], cập nhật alpha_j và alpha_i, sau đó cập nhật bias b. Project triển khai phiên bản giáo dục, ưu tiên khả năng đọc và kiểm chứng hơn tốc độ thư viện.")
    add_bullets(document, [
        "Chọn i đang vi phạm KKT và chọn j theo độ chênh lệch lỗi.",
        "Tính eta, giới hạn L/H và alpha mới.",
        "Cập nhật b bằng b1/b2 dựa trên alpha nằm trong khoảng mở.",
        "Lặp đến khi số pass không thay đổi đạt ngưỡng hoặc chạm max_iter.",
    ])
    add_figure(document, DIAGRAMS / "smo_pair_update.png", "Hình 3.2. Một vòng cập nhật cặp của SMO.", width=6.1)

    add_heading(document, "CHƯƠNG 4. CÀI ĐẶT VÀ THIẾT KẾ THỰC NGHIỆM", level=1)
    add_heading(document, "4.1. Kiến trúc project", level=2)
    document.add_paragraph("Các module math/ chứa geometry, primal, dual và KKT; optimization/smo.py chứa SMOOptimizer; models/kernel.py và models/svm.py chứa kernel và KernelSVM; evaluation.py và visualization.py dùng để đánh giá và xuất hình. Script 06_report_benchmark.py là entry point tái lập toàn bộ kết quả trong báo cáo.")
    add_figure(document, DIAGRAMS / "architecture.png", "Hình 4.1. Kiến trúc các module trong project.", width=6.1)
    add_heading(document, "4.2. Ví dụ giải tay", level=2)
    document.add_paragraph("Với bộ dữ liệu toy bốn điểm, nghiệm giải tích thu được w=[0.5,0.5], b=0, alpha=[0.25,0.25,0,0], margin width ≈2.828, primal objective = dual objective = 0.25 và KKT được thỏa mãn. SMO tái tạo cùng nghiệm với sai số số học ở bias.")
    add_equation(document, r"P=\frac{1}{2}\|w\|^2=0.25\;=\;D", "(4.1)")
    add_heading(document, "4.3. Cấu hình benchmark", level=2)
    add_figure(document, DIAGRAMS / "experiment_pipeline.png", "Hình 4.2. Pipeline thực nghiệm Banknote.", width=6.1)
    add_caption(document, "Bảng 4.1. Cấu hình benchmark và giao thức đánh giá.")
    add_table(document, ["Thành phần", "Cấu hình"], [
        ("Dataset", "UCI Banknote Authentication, 1.372 mẫu, 4 features, 2 lớp"),
        ("Split", "80/20, stratify, random_state=42"),
        ("Preprocessing", "StandardScaler fit trên train; test chỉ transform"),
        ("CV", "Stratified 5-fold, shuffle=True, random_state=42"),
        ("RBF grid", "C ∈ {0.1,1,10,100}; gamma ∈ {0.001,0.01,0.1,1}"),
        ("Custom SMO", "Linear kernel, C=1, tol=1e−3, max_passes=10, max_iter=100000"),
    ], widths=[1.7, 4.3])

    add_heading(document, "CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM", level=1)
    add_heading(document, "5.1. Dữ liệu và giao thức đánh giá", level=2)
    document.add_paragraph(f"Bộ dữ liệu có {summary['n_samples']} mẫu và {summary['n_features']} đặc trưng: {', '.join(summary['feature_names'])}. Sau khi chia, train có {summary['train_samples']} mẫu và test có {summary['test_samples']} mẫu. Nhãn dương được quy ước là +1; mọi metrics đều tính trên test chưa từng được dùng để chọn siêu tham số.")
    add_table(document, ["Đại lượng", "Giá trị"], [
        ("Tổng số mẫu", summary["n_samples"]),
        ("Số đặc trưng", summary["n_features"]),
        ("Train/Test", f"{summary['train_samples']} / {summary['test_samples']}"),
        ("Class -1 / +1", f"{summary['class_counts'].get('-1', '—')} / {summary['class_counts'].get('1', '—')}"),
        ("Random state", summary["random_state"]),
    ], widths=[2.4, 3.6])
    add_heading(document, "5.2. So sánh metrics", level=2)
    result_rows = []
    for _, row in metrics.iterrows():
        result_rows.append((row["model"], fmt(row["accuracy"]), fmt(row["precision"]), fmt(row["recall"]), fmt(row["f1"]), fmt(row["roc_auc"]), fmt(row["n_support_vectors"], 0)))
    add_caption(document, "Bảng 5.1. Metrics của các mô hình trên tập test.")
    add_table(document, ["Mô hình", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "SV"], result_rows, widths=[1.55, .72, .72, .72, .72, .8, .55])
    add_figure(document, FIGURES / "benchmark_metrics.png", "Hình 5.1. So sánh Accuracy, Precision, Recall và F1.", width=6.2)
    add_figure(document, FIGURES / "benchmark_confusion_matrices.png", "Hình 5.2. Ma trận nhầm lẫn trên tập test.", width=6.2)
    document.add_paragraph("RBF SVC sau tuning đạt kết quả tốt nhất trên split cố định. Custom SMO Linear khớp Linear SVC về metrics, cho thấy phần cài đặt Dual/SMO tái hiện đúng nghiệm phân loại trong cấu hình này; thời gian chạy và số vòng lặp vẫn lớn hơn thư viện tối ưu hóa chuyên dụng.")
    add_heading(document, "5.3. Ảnh hưởng của C và gamma", level=2)
    best = artifacts["best_rbf_params"]
    document.add_paragraph(f"Grid search chọn cấu hình {best}. Việc chọn được thực hiện bằng F1 trung bình trên 5-fold của train, không dùng test. Bảng/heatmap dưới đây cho thấy vùng tham số ảnh hưởng trực tiếp đến chất lượng nghiệm và nguy cơ quá khớp.")
    add_table(document, ["Tham số", "Giá trị tốt nhất"], [("C", best.get("svc__C")), ("gamma", best.get("svc__gamma"))], widths=[2, 4])
    add_figure(document, FIGURES / "rbf_grid_search_heatmap.png", "Hình 5.3. Mean F1 của RBF SVC trên 5-fold CV.", width=5.4)
    add_heading(document, "5.4. Hội tụ và trực quan hóa", level=2)
    add_figure(document, FIGURES / "custom_smo_convergence.png", "Hình 5.4. Lịch sử objective Dual của Custom SMO Linear.", width=6.0)
    add_figure(document, FIGURES / "decision_boundaries_2d.png", "Hình 5.5. Decision boundary trên hai đặc trưng đầu; đây là hình minh họa, không thay thế benchmark đủ 4 đặc trưng.", width=6.2)
    add_figure(document, FIGURES / "benchmark_roc.png", "Hình 5.6. ROC của các mô hình có decision score trên test.", width=6.0)

    add_heading(document, "CHƯƠNG 6. THẢO LUẬN VÀ HẠN CHẾ", level=1)
    add_heading(document, "6.1. Ý nghĩa tối ưu hóa", level=2)
    document.add_paragraph("Kết quả toy xác nhận đẳng thức primal–dual và KKT ở trường hợp có nghiệm giải tích. Trên Banknote, custom SMO đạt cùng metrics với Linear SVC trong split đã chọn và dual feasibility được kiểm tra bằng equality constraint cùng box constraints. Đây là bằng chứng về tính đúng của pipeline, không phải tuyên bố custom solver có hiệu năng tương đương LIBSVM.")
    add_heading(document, "6.2. Hạn chế", level=2)
    add_bullets(document, [
        "Custom SMO dùng chiến lược chọn cặp đơn giản và chưa hướng tới dataset lớn.",
        "Benchmark dùng một split cố định để trình bày minh bạch; kết luận rộng hơn cần lặp nhiều split hoặc nested CV.",
        "Boundary 2D chỉ dùng hai đặc trưng đầu để trực quan hóa; mô hình chính dùng đủ bốn đặc trưng.",
    ])
    add_heading(document, "KẾT LUẬN VÀ KIẾN NGHỊ", level=1)
    document.add_paragraph("Báo cáo đã kết nối hình học SVM với Primal, Lagrangian, Dual, KKT và SMO; triển khai solver giáo dục; và kiểm chứng bằng benchmark có preprocessing, CV, test metrics và biểu đồ. Hướng phát triển gồm cải thiện heuristic chọn working pair, hỗ trợ nhiều lớp, thêm repeated/nested CV và tối ưu bộ nhớ kernel.")

    add_heading(document, "TÀI LIỆU THAM KHẢO", level=1)
    refs = [
        "Cortes, C. & Vapnik, V. (1995). Support-Vector Networks. Machine Learning, 20, 273–297.",
        "Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. MSR-TR-98-14.",
        "Chang, C.-C. & Lin, C.-J. (2011). LIBSVM: A Library for Support Vector Machines.",
        "scikit-learn documentation — Support Vector Machines: https://scikit-learn.org/stable/modules/svm.html",
        "UCI Banknote Authentication: https://archive.ics.uci.edu/dataset/267/banknote+authentication",
    ]
    for index, ref in enumerate(refs, start=1):
        p = document.add_paragraph(style="Content")
        p.add_run(f"{index}. {ref}")

    add_heading(document, "PHỤ LỤC. CÁCH TÁI LẬP KẾT QUẢ", level=1)
    document.add_paragraph("Từ thư mục gốc project, cài requirements rồi chạy:")
    p = document.add_paragraph()
    p.style = "No Spacing" if "No Spacing" in [s.name for s in document.styles] else "Content"
    p.add_run("python -m pytest -q\npython -m experiments.02_hand_example\npython -m experiments.06_report_benchmark\npython reports/generate_docx.py").font.name = "Consolas"
    document.add_paragraph("Các file số liệu nằm trong reports/data/, biểu đồ trong reports/figures/ và bản báo cáo cuối cùng tại reports/svm_optimization_report.docx.")

    # Remove orphaned image relationships left by rebuilding from an earlier report.
    used_ids = set(document._element.xpath("//@r:embed | //@r:link"))
    for rel_id, rel in list(document.part.rels.items()):
        if rel.reltype.endswith("/image") and rel_id not in used_ids:
            document.part.drop_rel(rel_id)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(OUTPUT))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
