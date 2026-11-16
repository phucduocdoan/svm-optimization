"""Kiểm tra độc lập phần toán của báo cáo DOCX.

Không dùng lại bộ nhận diện của ``report_math`` để bộ kiểm tra không "mù" cùng chỗ
với bộ sinh. Các bước:

1. LaTeX trong ``docs/*.md`` chuyển được sang OMML.
2. Văn bản thường (ngoài Word math, ngoài code) không còn ký hiệu toán: ký tự toán,
   tên chữ cái Hy Lạp, chữ cái đơn đứng riêng, số có dấu, lời gọi hàm, phép toán
   giữa các toán hạng ngắn, đánh dấu ``$``/backtick còn sót.
3. Công thức khối: căn giữa, dạng ``\\t<math>\\t(c.n)``, chỉ nằm trong chương có số,
   đánh số liên tục từ 1 trong từng chương; không có công thức khối thiếu số.
4. Nội dung bản nền (``report_content.json``) còn đủ và đúng thứ tự trong DOCX.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from itertools import pairwise
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from latex2mathml.converter import convert as latex_to_mathml
from mathml2omml import convert as mathml_to_omml

ROOT = Path(__file__).resolve().parents[2]
DOCX = ROOT / "reports/output/svm_optimization_report.docx"
CONTENT = ROOT / "reports/source/report_content.json"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
CODE_FONT = "Consolas"

LATEX_DELIMITED = re.compile(r"\$\$(.+?)\$\$|(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL)
LABEL = re.compile(r"\((\d+)\.(\d+)\)\s*$")

GREEK_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
# Tên Greek trùng âm tiết tiếng Việt chỉ bị coi là toán khi viết liền chỉ số/ngoặc.
VIETNAMESE_SYLLABLES = {"phi", "chi", "xi", "mu", "nu", "pi", "eta", "tau"}
LEFTOVER = [
    ("ký tự toán", re.compile(r"[_^=<>|≤≥≈≠±·×÷−∑Σ∏√∞∀∃∈∉⊂⊆∪∩⇒⇔→‖∂∇$`Ͱ-Ͽ]")),
    ("tên chữ Hy Lạp", re.compile(
        rf"(?<![\w-])(?:{'|'.join(n for n in GREEK_NAMES if n not in VIETNAMESE_SYLLABLES)})(?![\w-])"
        rf"|(?<![\w-])(?:{'|'.join(VIETNAMESE_SYLLABLES)})(?=[_(\d])", re.IGNORECASE)),
    # Chữ cái đơn đứng riêng; trừ tên viết tắt trong trích dẫn ("C. &", "C.-J. (")
    # và mạo từ tiếng Anh đầu tên tài liệu ("A Fast Algorithm").
    ("chữ cái đơn", re.compile(r"(?<![\w.&'’-])(?!(?:A|I) [A-Z])[A-Za-z](?![\w'’-])(?!\.(?:\s*(?:[&(,]|[A-Z]\.)|-))")),
    ("số có dấu", re.compile(r"(?:^|(?<=[\s(\[]))[-+]\d")),
    ("lời gọi hàm", re.compile(r"(?<![\w-])(?:[A-Za-z]|exp|max|min|log|sign|sqrt)\s?\(")),
    ("biến có chỉ số", re.compile(r"(?<![\w-])[a-z]\d+(?![\w-])")),
    ("phép toán", re.compile(r"(?<![\w.,/-])(?:\w|\d+(?:[.,]\d+)?)\s*[+*/]\s*(?:\w|\d+(?:[.,]\d+)?)(?![\w/])")),
]
URL = re.compile(r"https?://\S+")
# Dãy chữ cái theo bảng chữ cái ("A,B,C") là văn xuôi, không phải biến.
ALPHABET_RUN = re.compile(r"(?<!\w)[A-Z](?:\s*,\s*[A-Z])+(?!\w)")
# Từ trong bản nền được thay bằng ký hiệu khi chuyển sang công thức.
WORD_OPERATORS = {"khoảng", "Tổng", "căn", "bậc", "hai", "với", "mọi"}
TOKEN = re.compile(r"\d+|[^\W\d_]+")  # tách "2w" thành "2", "w" như trong OMML
# Bản nền viết tên chữ Hy Lạp bằng chữ Latin ("gamma"); DOCX dùng ký tự (γ).
GREEK_LETTER = {n: unicodedata.lookup(f"GREEK SMALL LETTER {n.upper().replace('LAMBDA', 'LAMDA')}") for n in GREEK_NAMES}
GREEK_VARIANT = str.maketrans("ϕϵϑϰϱϖ", "φεθκρπ")


def check_source_latex() -> int:
    count = 0
    for path in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "reports/source/report_additions.md"]:
        for match in LATEX_DELIMITED.finditer(path.read_text(encoding="utf-8")):
            formula = match.group(1) or match.group(2)
            # mathml2omml không hỗ trợ môi trường aligned.
            if r"\begin{aligned}" not in formula:
                try:
                    mathml_to_omml(latex_to_mathml(formula))
                except Exception as exc:
                    raise AssertionError(f"Không chuyển được LaTeX trong {path.name}: {formula[:120]!r}") from exc
            count += 1
    return count


def _in_math(element) -> bool:
    return any(a.tag == M + "oMath" for a in element.iterancestors())


def _is_code(t) -> bool:
    fonts = t.getparent().find(qn("w:rPr") + "/" + qn("w:rFonts"))
    return fonts is not None and fonts.get(qn("w:ascii")) == CODE_FONT


def plain_text(p) -> str:
    """Văn bản thường của đoạn; mỗi công thức/code thành một dấu ⟦⟧ để không dính chữ."""
    out = []
    for el in p.iter(qn("w:t"), qn("w:tab"), qn("w:br"), M + "oMath"):
        if el.tag == M + "oMath":
            out.append("⟦⟧")
        elif _in_math(el):
            continue
        elif el.tag == qn("w:br"):
            out.append("\n")
        elif el.tag == qn("w:tab"):
            if el.getparent().tag == qn("w:r"):  # bỏ tab stop trong w:pPr/w:tabs
                out.append("\t")
        elif _is_code(el):
            out.append("⟦⟧")
        else:
            out.append(el.text or "")
    return "".join(out)


def _alphabetical(run: str) -> bool:
    letters = [ord(c) for c in run if c.isalpha()]
    return all(b == a + 1 for a, b in pairwise(letters))


def check_leftovers(paragraphs) -> list[str]:
    errors = []
    for p in paragraphs:
        text = URL.sub("⟦⟧", plain_text(p))
        text = ALPHABET_RUN.sub(lambda m: m.group() if not _alphabetical(m.group()) else "⟦⟧", text)
        if LABEL.search(text) and p.find(".//" + M + "oMath") is not None:
            text = LABEL.sub("", text)
        for name, pattern in LEFTOVER:
            for m in pattern.finditer(text):
                errors.append(f"{name} {m.group()!r} chưa thành Word math: {text.strip()[:110]!r}")
    return errors


def _chapter(p) -> bool | None:
    """True: Heading 1 có số chương; False: Heading 1 không số; None: không phải chương."""
    style = p.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    if style is None or style.get(qn("w:val")) not in ("Heading1", "1"):
        return None
    num = p.find(qn("w:pPr") + "/" + qn("w:numPr") + "/" + qn("w:numId"))
    return not (num is not None and num.get(qn("w:val")) == "0")


def check_equations(body) -> tuple[list[str], list[str]]:
    errors, labels = [], []
    chapter, count = None, 0
    for p in body.iter(qn("w:p")):
        if p.getparent().tag != qn("w:body"):
            continue
        heading = _chapter(p)
        if heading is not None:
            chapter = (chapter or 0) + 1 if heading else None
            if heading:
                count = 0
            continue
        has_math = p.find(".//" + M + "oMath") is not None
        text = plain_text(p)
        label = LABEL.search(text)
        rest = LABEL.sub("", text).replace("⟦⟧", "").strip()
        display = has_math and (label or not rest)
        if not display:
            continue
        where = f"{text.strip()[:80]!r}"
        if label is None:
            errors.append(f"Công thức khối không có số: {where}")
            continue
        labels.append(label.group().strip())
        if chapter is None:
            errors.append(f"Công thức khối ngoài chương có số: {where}")
            continue
        count += 1
        if (int(label[1]), int(label[2])) != (chapter, count):
            errors.append(f"Số công thức {label.group().strip()} sai, cần ({chapter}.{count}): {where}")
        if re.sub(r"\s*\([^()]*\)\s*$|[,.;]\s*$", "", rest):
            errors.append(f"Đoạn công thức có chữ thừa ngoài ghi chú: {where}")
        if not re.fullmatch(r"\t⟦⟧.*\t\(\d+\.\d+\)\s*", text, re.DOTALL):
            errors.append(f"Công thức không theo dạng tab-công thức-tab-số: {where}")
        jc = p.find(qn("w:pPr") + "/" + qn("w:jc"))
        if jc is None or jc.get(qn("w:val")) != "center":
            errors.append(f"Công thức không căn giữa: {where}")
    return errors, labels


CAPTION_LABEL = re.compile(r"^(Hình|Bảng) (\d+)\.(\d+)\. \S")


def _style(p) -> str | None:
    style = p.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return None if style is None else style.get(qn("w:val"))


def check_captions(body) -> tuple[list[str], dict[str, list[str]]]:
    """Mỗi hình có chú thích ngay dưới, mỗi chú thích bảng đứng ngay trên bảng; số (c.n) liên tục theo chương; danh mục khớp."""
    errors, found = [], {"Hình": [], "Bảng": []}
    chapter, counts = None, {}
    for el in body:
        if el.tag == qn("w:p") and _chapter(el) is not None:
            chapter, counts = ((chapter or 0) + 1 if _chapter(el) else None), {}
        if el.findall(".//" + qn("w:drawing")):
            following = el.getnext()
            if following is None or _style(following) != "Caption" or not plain_text(following).startswith("Hình "):
                errors.append(f"Hình không có chú thích ngay dưới, sau đoạn: {plain_text(el.getprevious())[:80]!r}")
        if el.tag != qn("w:p") or _style(el) != "Caption":
            continue
        text = plain_text(el)
        label = CAPTION_LABEL.match(text)
        if label is None:
            errors.append(f"Chú thích thiếu nhãn Hình/Bảng c.n.: {text[:80]!r}")
            continue
        kind = label.group(1)
        counts[kind] = counts.get(kind, 0) + 1
        expected = (str(chapter), str(counts[kind]))
        if (label.group(2), label.group(3)) != expected:
            errors.append(f"Đánh số sai: {text[:40]!r}, cần {kind} {expected[0]}.{expected[1]}.")
        neighbour = el.getprevious() if kind == "Hình" else el.getnext()
        if kind == "Bảng" and (neighbour is None or neighbour.tag != qn("w:tbl")):
            errors.append(f"Chú thích bảng không đứng ngay trên bảng: {text[:80]!r}")
        if kind == "Hình" and (neighbour is None or not neighbour.findall(".//" + qn("w:drawing"))):
            errors.append(f"Chú thích hình không đứng ngay dưới hình: {text[:80]!r}")
        found[kind].append(text)
    for kind, title in (("Hình", "DANH MỤC CÁC HÌNH ẢNH"), ("Bảng", "DANH MỤC BẢNG BIỂU")):
        heading = next((p for p in body.iter(qn("w:p")) if plain_text(p).strip() == title), None)
        listed = plain_text(heading.getnext()).split("\n") if heading is not None else []
        if listed != found[kind]:
            errors.append(f"{title} không khớp chú thích trong bài: {len(listed)} dòng, {len(found[kind])} chú thích.")
    return errors, found


def _doc_tokens(body) -> list[str]:
    tokens = []
    for el in body.iter(qn("w:t"), M + "t", M + "chr"):
        value = el.get(M + "val") if el.tag == M + "chr" else el.text
        tokens += TOKEN.findall((value or "").translate(GREEK_VARIANT))
    return tokens


def _joined(doc: list[str], word: str, j: int, limit: int) -> int | None:
    """Một từ bị tách thành nhiều token liền nhau: run "C"+"ực", OMML "h","i","n","g","e"."""
    joined = ""
    while j < limit and word.startswith(joined + doc[j]):
        joined, j = joined + doc[j], j + 1
        if joined == word:
            return j
    return None


def _match(doc: list[str], words: list[str], start: int) -> int | None:
    """Khớp ``words`` vào ``doc`` từ vị trí ``start`` trong một cửa sổ ngắn; trả về vị trí kết thúc."""
    i, j, limit = 0, start, min(len(doc), start + len(words) + 12)
    while i < len(words) and j < limit:
        if doc[j] == words[i] or doc[j] == GREEK_LETTER.get(words[i]):
            i, j = i + 1, j + 1
        elif (k := _joined(doc, words[i], j, limit)) is not None:
            i, j = i + 1, k
        elif i and words[i] in WORD_OPERATORS:
            i += 1
        else:
            j += 1
    return j if i == len(words) else None


def check_content(body, records) -> list[str]:
    """Từ của mỗi đoạn bản nền phải xuất hiện theo thứ tự, gần liền nhau, trong DOCX."""
    doc, errors, pos = _doc_tokens(body), [], 0
    for record in records:
        text = LABEL.sub("", record["text"]).replace("\uf0b7", "")
        words = TOKEN.findall(text)
        if not words:
            continue
        end = next((e for s in range(pos, len(doc)) if words[0].startswith(doc[s])
                    and (e := _match(doc, words, s)) is not None), None)
        if end is None:
            errors.append(f"Nội dung bản nền bị mất hoặc đổi thứ tự: {record['text'][:100]!r}")
        else:
            pos = end
    return errors


# Word vẽ ô vuông giữ chỗ cho ô bắt buộc bị thiếu hoặc rỗng, trừ khi ô đó được ẩn (subHide/supHide/degHide).
SLOTS = {"nary": ("sub", "sup", "e"), "rad": ("deg", "e"), "f": ("num", "den"), "sSub": ("e", "sub"),
         "sSup": ("e", "sup"), "sSubSup": ("e", "sub", "sup"), "func": ("fName", "e"),
         "limLow": ("e", "lim"), "limUpp": ("e", "lim")}


def check_placeholders(body) -> list[str]:
    errors = []
    for kind, slots in SLOTS.items():
        for el in body.iter(M + kind):
            for slot in slots:
                content = el.find(M + slot)
                if content is not None and "".join(content.itertext()).strip():
                    continue
                flag = el.find(f"{M}{kind}Pr/{M}{slot}Hide")
                if flag is None or flag.get(M + "val") in ("0", "off", "false"):
                    math = el.xpath("ancestor::m:oMath[1]", namespaces={"m": M[1:-1]})
                    errors.append(f"Ô {slot} của {kind} trống sẽ hiện ô vuông: {''.join(math[0].itertext())[:80]!r}")
    return errors


def main() -> None:
    latex_count = check_source_latex()
    document = Document(DOCX)
    body = document.element.body
    paragraphs = [p for p in body.iter(qn("w:p"))]
    errors = check_leftovers(paragraphs)
    equation_errors, labels = check_equations(body)
    errors += equation_errors
    errors += check_placeholders(body)
    caption_errors, captions = check_captions(body)
    errors += caption_errors
    records = json.loads(CONTENT.read_text(encoding="utf-8"))
    errors += check_content(body, records)
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} lỗi.")
    math_count = len(body.findall(".//" + M + "oMath"))
    print(f"OK: {latex_count} công thức LaTeX trong nguồn Markdown; {math_count} Word math; "
          f"{len(labels)} công thức khối {labels[0]}…{labels[-1]}; {len(captions['Hình'])} hình, "
          f"{len(captions['Bảng'])} bảng đánh số đúng; {len(records)} đoạn bản nền còn đủ.")


if __name__ == "__main__":
    main()
