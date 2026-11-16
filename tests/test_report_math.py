import check_math
import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from report_math import find_math, omml, render_math, to_latex

PROSE = [
    "phi tuyến", "chi phí", "xi măng", "k-Fold", "5-fold", "F1-Score", "COVID-19", "C-SVM", "one-vs-one",
    "AT&T", "RBF / Gaussian", "Cortes, C. & Vapnik, V. (1995)", "Platt, J. C. (1998)", "theo (2.2)",
    "Chương 2", "95%", "pp. 1-2", "A, B, C", "A Fast Algorithm",
]

MATH = {
    "y_i . (w^T . x_i - b) >= 1": r"y_{i} (w^{T} x_{i}-b) \geq 1",
    "(1/2) . ||w||^2 + C . Tổng(xi_i)": r"\frac{1}{2} \|w\|^{2}+C \sum_{i}\xi_{i}",
    "L_hinge = max(0, 1 - y_i . (w^T . x_i - b))": r"L_{\mathrm{hinge}} = \max(0,1-y_{i} (w^{T} x_{i}-b))",
    "K(K-1)/2": r"\frac{K(K-1)}{2}",
    "x_i-x_j": r"x_{i}-x_{j}",
    "0 <= alpha_i <= C": r"0 \leq \alpha_{i} \leq C",
    "exp(-gamma . ||x - z||^2)": r"\exp(-\gamma \|x-z\|^{2})",
    "căn bậc hai(2)": r"\sqrt{2}",
}


@pytest.mark.parametrize("text", PROSE)
def test_prose_is_not_math(text):
    assert find_math(text) == []


@pytest.mark.parametrize("text, latex", MATH.items())
def test_math_is_detected_as_one_expression(text, latex):
    assert to_latex(text) == latex


def test_math_inside_sentence_keeps_surrounding_prose():
    [span] = find_math("cần K(K-1)/2 bộ phân loại")
    assert span.latex == r"\frac{K(K-1)}{2}"


def _report():
    document = Document()
    document.add_heading("Mở đầu", 1)
    unnumbered = document.paragraphs[0]._p.get_or_add_pPr()
    num_pr, num_id = OxmlElement("w:numPr"), OxmlElement("w:numId")
    num_id.set(qn("w:val"), "0")
    num_pr.append(num_id)
    unnumbered.append(num_pr)
    document.add_paragraph("Lời mở đầu có y_i . (w^T . x_i - b) >= 1.")
    document.add_heading("Cơ sở lý thuyết", 1)
    document.add_paragraph("Cực tiểu hóa hàm mục tiêu: (1/2) . ||w||^2 (2.2)")
    document.add_paragraph("Ta có f(x) = w^T . x + b là siêu phẳng, với K(K-1)/2 bộ phân loại.")
    document.add_heading("Thực nghiệm", 1)
    document.add_paragraph("Kernel: K(x, z) = exp(-gamma . ||x - z||^2) (công thức RBF)")
    render_math(document)
    return document


def _texts(document):
    return [check_math.plain_text(p._p) for p in document.paragraphs]


def test_render_math_numbers_and_centers_display_equations_by_chapter():
    document = _report()
    assert _texts(document) == [
        "Mở đầu", "Lời mở đầu có ⟦⟧.",
        "Cơ sở lý thuyết", "Cực tiểu hóa hàm mục tiêu:", "\t⟦⟧\t(1.1)",
        "Ta có", "\t⟦⟧\t(1.2)", "là siêu phẳng, với ⟦⟧ bộ phân loại.",
        "Thực nghiệm", "Kernel:", "\t⟦⟧ (công thức RBF)\t(2.1)",
    ]
    body = document.element.body
    assert check_math.check_equations(body) == ([], ["(1.1)", "(1.2)", "(2.1)"])
    assert check_math.check_leftovers(body.iter(qn("w:p"))) == []


def test_inline_fraction_is_linear_display_fraction_is_stacked():
    document = _report()
    fractions = {p.text: [f.find(f"{check_math.M}fPr") is not None for f in p._p.iter(f"{check_math.M}f")]
                 for p in document.paragraphs}
    assert fractions["\t\t(1.1)"] == [False]
    assert [linear for linear in fractions.values() if linear == [True]] == [[True]]


def test_sum_without_upper_limit_hides_empty_slot():
    body = OxmlElement("w:body")
    for latex in (r"\sum_i \alpha_i y_i", r"\sum_{i=1}^{n} \xi_i"):
        body.append(omml(latex))
    assert check_math.check_placeholders(body) == []
    hides = [[flag.tag[len(check_math.M):] for flag in nary.find(f"{check_math.M}naryPr")
              if flag.tag.endswith("Hide")] for nary in body.iter(f"{check_math.M}nary")]
    assert hides == [["supHide"], []]
    for flag in body.iter(f"{check_math.M}supHide"):
        flag.getparent().remove(flag)
    assert len(check_math.check_placeholders(body)) == 1


def test_checker_catches_leftover_math_wrong_number_and_lost_content():
    document = _report()
    body = document.element.body
    document.paragraphs[1].add_run(" và x_i chưa đổi")
    assert check_math.check_leftovers(body.iter(qn("w:p")))
    label = document.paragraphs[6].runs[-1]
    label.text = "\t(1.3)"
    errors, _ = check_math.check_equations(body)
    assert any("(1.3)" in e for e in errors)
    records = [{"text": "Cực tiểu hóa hàm mục tiêu: (1/2) . ||w||^2 (2.2)"},
               {"text": "Kernel: K(x, z) = exp(-gamma . ||x - z||^2)"}]
    assert check_math.check_content(body, records) == []
    assert check_math.check_content(body, records[::-1])
    assert check_math.check_content(body, [{"text": "Một câu không có trong báo cáo"}])
