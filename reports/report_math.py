r"""Nhận diện ký hiệu toán viết dạng chữ trong văn bản báo cáo và chuyển sang LaTeX.

Không có danh sách công thức cố định. Văn bản được tách thành token theo các
quy tắc cơ sở (biến, số, toán tử, quan hệ, ngoặc, chuẩn ``||.||``, lời gọi hàm),
sau đó một parser đệ quy nhận ra mọi biểu thức đúng cú pháp và sinh LaTeX.

Văn bản do generator tự viết có thể đánh dấu tường minh như Markdown:
``$...$`` (công thức trong dòng), ``$$...$$`` (công thức khối), ```code```.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise

from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from latex2mathml.converter import convert as latex_to_mathml
from mathml2omml import convert as mathml_to_omml

# --- Từ vựng cơ sở --------------------------------------------------------

GREEK = {name: "\\" + name for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"]}
# Tên Greek trùng âm tiết tiếng Việt ("phi tuyến", "chi phí", "xi măng"):
# chỉ là toán khi có chỉ số, được gọi như hàm hoặc nằm trong biểu thức.
AMBIGUOUS_GREEK = {"phi", "chi", "xi", "mu", "nu", "pi", "eta", "tau"}
FUNCTIONS = {name: "\\" + name for name in ["max", "min", "exp", "log", "ln", "sin", "cos", "tan", "tanh", "arg", "sup", "inf", "det"]}
FUNCTIONS |= {"sign": r"\operatorname{sign}", "argmax": r"\arg\max", "argmin": r"\arg\min", "sqrt": "sqrt", "sum": "sum"}
# Hàm và toán tử viết bằng tiếng Việt trong bản nền.
WORD_FUNCTIONS = {"căn bậc hai": "sqrt", "Tổng": "sum"}
WORD_RELATIONS = {"khoảng": r"\approx"}
FORALL_WORDS = ("với mọi",)
RELATIONS = {">=": r"\geq", "<=": r"\leq", "=>": r"\Rightarrow", "<=>": r"\Leftrightarrow", "!=": r"\neq",
             "=": "=", "<": "<", ">": ">", "≥": r"\geq", "≤": r"\leq", "≈": r"\approx", "≠": r"\neq",
             "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow", "→": r"\to"}
ADDITIVE = {"+": "+", "-": "-", "−": "-"}
MULTIPLICATIVE = {".": "dot", "*": "dot", "·": "dot", "×": "dot", "/": "div"}

# --- Tokenizer -------------------------------------------------------------

def _alternation(words) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


TOKEN = re.compile(rf"""
    (?P<space>\s+)
  | (?P<url>https?://\S+)
  | (?P<initials>(?<![\w.])[A-Z]\.(?:-[A-Z]\.)*(?=\s*(?:[&(,]|[A-Z]\.)))      # tên viết tắt trong trích dẫn
  | (?P<wordfunc>(?:{_alternation(WORD_FUNCTIONS)})(?=\())
  | (?P<forall>(?<!\w)(?:{_alternation(FORALL_WORDS)})(?!\w))
  | (?P<wordrel>(?<!\w)(?:{_alternation(WORD_RELATIONS)})(?!\w))
  | (?P<rel>{_alternation(RELATIONS)})
  | (?P<norm>\|\|)
  | (?P<number>\d+(?:[.,]\d+)?(?![\w]|-\w))
  | (?P<word>[\w&]+(?:-[\w&]+)*)
  | (?P<ellipsis>\.\.\.|…)
  | (?P<dot>(?<=\s)\.(?=\s))
  | (?P<op>[-+*/^·×−])
  | (?P<open>[(\[{{])
  | (?P<close>[)\]}}])
  | (?P<comma>,)
  | (?P<other>.)
""", re.VERBOSE | re.DOTALL)

IDENT = re.compile(r"(?P<coef>\d*)(?P<base>[A-Za-z]+)(?:_(?P<sub>[A-Za-z0-9]+))?")


@dataclass
class Token:
    kind: str
    text: str
    start: int
    end: int
    space_before: bool = False


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    space = True
    for m in TOKEN.finditer(text):
        kind = m.lastgroup
        if kind == "space":
            space = True
            continue
        if kind == "word" and _subtraction(m.group()):
            start = m.start()
            for i, piece in enumerate(m.group().split("-")):
                if i:
                    tokens.append(Token("op", "-", start - 1, start))
                kind = "number" if piece.isdigit() else _classify_word(piece, text, start + len(piece))
                tokens.append(Token(kind, piece, start,
                                    start + len(piece), space and not i))
                start += len(piece) + 1
        else:
            if kind == "word":
                kind = _classify_word(m.group(), text, m.end())
            tokens.append(Token(kind, m.group(), m.start(), m.end(), space))
        space = False
    return tokens


def _subtraction(word: str) -> bool:
    """"K-1", "x_i-x_j" là phép trừ; "k-Fold", "COVID-19", "5-fold", "1-2" là chữ."""
    pieces = word.split("-")
    symbol = [p.isdigit() or bool((m := IDENT.fullmatch(p)) and not m["coef"] and (m["sub"] or len(m["base"]) == 1))
              for p in pieces]
    return len(pieces) > 1 and all(symbol) and not all(p.isdigit() for p in pieces)


def _classify_word(word: str, text: str, end: int) -> str:
    """var / weakvar (Greek trùng tiếng Việt) / func / prose."""
    called = text[end:end + 1] == "("
    if len(word) == 1 and word.isupper() and _letter_prose(word, text, end):
        return "prose"
    m = IDENT.fullmatch(word)
    if not m:
        return "prose"
    base, sub, coef = m["base"], m["sub"], m["coef"]
    if coef and not sub:          # "2D" là chữ; "2w_1" là hệ số nhân biến
        return "prose"
    if called and not sub and (base in FUNCTIONS or base in GREEK or len(base) == 1):
        return "func"
    if len(base) == 1 or (base in GREEK and (sub or base not in AMBIGUOUS_GREEK)):
        return "var"
    if base in GREEK:
        return "weakvar"
    return "prose"


def _letter_prose(letter: str, text: str, end: int) -> bool:
    """Chữ hoa đứng riêng nhưng không phải ký hiệu: mạo từ tiếng Anh ("A Fast
    Algorithm") hoặc liệt kê theo bảng chữ cái ("A, B, C")."""
    start = end - 1
    if letter in "AI" and re.match(r" [A-Z][a-z]", text[end:]):
        return True
    following = re.match(r"\s*,\s*(\w)\b", text[end:])
    preceding = re.search(r"\b(\w)\s*,\s*$", text[:start])
    return bool((following and ord(following[1]) == ord(letter) + 1)
                or (preceding and ord(preceding[1]) == ord(letter) - 1))


# --- Parser ------------------------------------------------------------------

@dataclass
class Node:
    latex: str
    strong: bool          # chắc chắn là toán (có biến, hàm, chuẩn, toán tử...)
    kind: str = "atom"    # atom | group | frac | number | signed
    operands: int = 1     # số toán hạng cấp ngoài cùng (để phân loại công thức khối)
    relation: str = ""    # quan hệ cấp ngoài cùng đầu tiên, nếu có
    inner: str = ""       # nội dung trong ngoặc của một nhóm
    compound: bool = False  # có phép toán hoặc lời gọi hàm (không chỉ là ký hiệu/giá trị)


class Parser:
    def __init__(self, tokens: list[Token]):
        self.t = tokens
        self.i = 0

    def peek(self, offset: int = 0) -> Token | None:
        j = self.i + offset
        return self.t[j] if j < len(self.t) else None

    def at(self, kind: str, text: str | None = None, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok is not None and tok.kind == kind and (text is None or tok.text == text)

    def attempt(self, rule):
        saved = self.i
        try:
            node = rule()
        except SyntaxError:
            node = None
        if node is None:
            self.i = saved
        return node

    # statement := expr (rel expr)* [", với mọi" statement] [list tail]
    def statement(self, allow_lists: bool = True) -> Node:
        left = self.expr()
        parts, strong, operands, relation = [left.latex], left.strong, left.operands, ""
        compound = left.compound
        while True:
            tok = self.peek()
            if tok and tok.kind in ("rel", "wordrel"):
                saved = self.i
                self.i += 1
                right = self.attempt(self.expr)
                if right is None:
                    self.i = saved
                    break
                symbol = RELATIONS.get(tok.text) or WORD_RELATIONS[tok.text]
                relation = relation or symbol
                parts.append(f" {symbol} {right.latex}")
                strong = True if (left.strong or right.strong or tok.kind == "rel") else strong
                operands += right.operands
                compound = compound or right.compound
                continue
            if tok and tok.kind == "comma" and self.at("forall", offset=1):
                saved = self.i
                self.i += 2
                rest = self.attempt(self.statement)
                if rest is None:
                    self.i = saved
                    break
                parts.append(r",\text{ }\forall " + rest.latex)
                strong = True
                continue
            if allow_lists and relation and tok and tok.kind == "comma":
                tail = self.attempt(self.list_tail)
                if tail:
                    parts.append(tail)
                    continue
            break
        if len(parts) == 1:
            return left
        return Node("".join(parts), strong, "atom", operands, relation, compound=compound)

    def list_tail(self) -> str:
        """", 2, ..., N" sau một quan hệ; chỉ nhận khi có dấu ba chấm."""
        items = []
        while self.at("comma") and self.peek(1) and self.peek(1).kind in ("number", "var", "ellipsis"):
            self.i += 1
            tok = self.peek()
            self.i += 1
            items.append(r"\ldots" if tok.kind == "ellipsis" else self._leaf(tok).latex)
        if r"\ldots" not in items:
            raise SyntaxError
        return "," + ",".join(items)

    def expr(self) -> Node:
        node = self.term()
        latex, strong, operands = node.latex, node.strong, 1
        while self.at("op") and self.peek().text in ADDITIVE and self.peek(1) is not None:
            saved = self.i
            op = ADDITIVE[self.peek().text]
            self.i += 1
            right = self.attempt(self.term)
            if right is None:
                self.i = saved
                break
            latex += f"{op}{right.latex}"
            strong = True
            operands += 1
        if operands == 1:
            return node
        return Node(latex, strong, "atom", operands, compound=True)

    def _product_op(self) -> str | None:
        tok = self.peek()
        if tok is None:
            return None
        if tok.kind == "dot":
            return "dot"
        if tok.kind == "op" and tok.text in MULTIPLICATIVE:
            return MULTIPLICATIVE[tok.text]
        return None

    def term(self) -> Node:
        node = self.factor()
        operands = 1
        while (op := self._product_op()):
            saved = self.i
            self.i += 1
            right = self.attempt(self.factor)
            if right is None:
                self.i = saved
                break
            operands += 1
            if op == "div":
                node = Node(rf"\frac{{{_unwrap(node)}}}{{{_unwrap(right)}}}", True, "frac", compound=True)
            else:
                numeric_left = node.kind in ("number", "signed")
                joiner = r" \cdot " if right.latex[:1] in "0123456789+-" or (numeric_left and right.kind == "group") else " "
                node = Node(node.latex.rstrip() + joiner + right.latex, True, "atom", compound=True)
        node.operands = operands if operands > 1 else node.operands
        return node

    def factor(self) -> Node:
        tok = self.peek()
        if tok and tok.kind == "op" and tok.text in ADDITIVE:
            nxt = self.peek(1)
            if nxt is None or nxt.space_before:
                raise SyntaxError
            self.i += 1
            inner = self.power()
            return Node(ADDITIVE[tok.text] + inner.latex, True, "signed" if inner.kind == "number" else "atom",
                        compound=inner.compound)
        return self.power()

    def power(self) -> Node:
        node = self.primary()
        while self.at("op", "^") and self.peek(1) and not self.peek(1).space_before:
            self.i += 1
            exponent = self.primary()
            node = Node(f"{node.latex}^{{{_unwrap(exponent)}}}", True, "atom", compound=node.compound)
        return node

    def primary(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise SyntaxError
        if tok.kind in ("number", "var", "weakvar"):
            self.i += 1
            return self._leaf(tok)
        if tok.kind == "norm":
            self.i += 1
            inner = self.expr()
            self._expect("norm")
            return Node(rf"\|{inner.latex}\|", True, "group", compound=inner.compound)
        if tok.kind == "open":
            return self.group()
        if tok.kind in ("func", "wordfunc"):
            self.i += 1
            name = FUNCTIONS.get(tok.text) or WORD_FUNCTIONS.get(tok.text) or _ident_latex(tok.text).rstrip()
            args = self.group()
            inner = args.inner
            if name == "sqrt":
                return Node(rf"\sqrt{{{inner}}}", True, "group", compound=True)
            if name == "sum":
                index = re.fullmatch(r"[^_]+_\{([a-z])\}", inner)
                return Node((rf"\sum_{{{index[1]}}}" if index else r"\sum") + inner, True, "atom", compound=True)
            return Node(name + args.latex, True, "group", compound=True)
        raise SyntaxError

    def group(self) -> Node:
        open_tok = self.peek()
        self.i += 1
        items = [self.statement(allow_lists=True)]
        while self.at("comma"):
            self.i += 1
            if self.at("ellipsis"):
                self.i += 1
                items.append(Node(r"\ldots", True))
            else:
                items.append(self.statement())
        close = {"(": ")", "[": "]", "{": "}"}[open_tok.text]
        self._expect("close", close)
        body = ",".join(item.latex for item in items)
        strong = any(item.strong for item in items) or (open_tok.text in "[{" and len(items) > 1)
        compound = any(item.compound for item in items)
        if open_tok.text == "(" and len(items) == 1 and items[0].kind == "frac":
            return Node(body, strong, "frac", inner=body, compound=True)
        tall = any(cmd in body for cmd in (r"\frac", r"\sum", r"\sqrt"))
        left, right = {"(": (r"\left(", r"\right)") if tall else ("(", ")"), "[": ("[", "]"), "{": (r"\{", r"\}")}[open_tok.text]
        return Node(f"{left}{body}{right}", strong, "group", inner=body, compound=compound)

    def _expect(self, kind: str, text: str | None = None) -> None:
        if not self.at(kind, text):
            raise SyntaxError
        self.i += 1

    @staticmethod
    def _leaf(tok: Token) -> Node:
        if tok.kind == "number":
            return Node(tok.text.replace(",", "{,}"), False, "number")
        return Node(_ident_latex(tok.text), tok.kind == "var")


def _ident_latex(word: str) -> str:
    m = IDENT.fullmatch(word)
    base = GREEK.get(m["base"], m["base"])
    sub = m["sub"]
    if sub:
        sub = rf"\mathrm{{{sub}}}" if len(sub) > 1 and sub.isalpha() else sub
        return f"{m['coef']}{base}_{{{sub}}}"
    return m["coef"] + base + (" " if base.startswith("\\") else "")


def _unwrap(node: Node) -> str:
    return node.inner if node.kind == "group" and node.latex.startswith(("(", r"\left(")) else node.latex


# --- API ---------------------------------------------------------------------

@dataclass
class MathSpan:
    start: int
    end: int
    latex: str
    operands: int = 1
    relation: str = ""
    compound: bool = False
    explicit: str = ""     # "" | "inline" | "display" | "code"


EXPLICIT = re.compile(r"\$\$(?P<display>.+?)\$\$|\$(?P<inline>[^$]+)\$|`(?P<code>[^`]+)`", re.DOTALL)


def find_math(text: str) -> list[MathSpan]:
    """Mọi đoạn toán trong ``text``: đánh dấu tường minh trước, sau đó tự nhận diện."""
    spans: list[MathSpan] = []
    cursor = 0
    for m in EXPLICIT.finditer(text):
        spans += _detect(text, cursor, m.start())
        kind = m.lastgroup
        spans.append(MathSpan(m.start(), m.end(), m.group(kind).strip(), explicit=kind))
        cursor = m.end()
    return spans + _detect(text, cursor, len(text))


def _detect(text: str, lo: int, hi: int) -> list[MathSpan]:
    tokens = [t for t in tokenize(text[lo:hi])]
    spans = []
    i = 0
    while i < len(tokens):
        parser = Parser(tokens)
        parser.i = i
        node = parser.attempt(parser.statement)
        if node is not None and node.strong and parser.i > i:
            first, last = tokens[i], tokens[parser.i - 1]
            spans.append(MathSpan(lo + first.start, lo + last.end, node.latex.strip(), node.operands, node.relation,
                                  node.compound))
            i = parser.i
        else:
            i += 1
    return spans


def to_latex(text: str) -> str:
    """LaTeX của toàn bộ ``text`` nếu nó là đúng một biểu thức toán."""
    spans = find_math(text)
    if len(spans) != 1 or (spans[0].start, spans[0].end) != (0, len(text.strip())):
        raise ValueError(f"Không phải một biểu thức toán: {text!r}")
    return spans[0].latex


# --- Word ------------------------------------------------------------------
# Công thức khối: (1) ``$$...$$``; (2) cả đoạn chỉ là toán, có thể có câu dẫn
# kết thúc bằng ":", nhãn cũ "(2.2)" và một chú thích trong ngoặc ở cuối;
# (3) một phương trình "=" có phép toán nằm giữa câu. Còn lại là toán trong dòng.

OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
CODE_FONT = "Consolas"
LABEL = re.compile(r"\s*\(\d+(?:\.\d+)+\)\s*$")
REMARK = re.compile(r"\s*\([^()]*\)\s*$")
CONNECTORS = {"": "", ",": r",\text{ }", "và": r"\text{ và }", ", và": r",\text{ và }"}
PROSE_STYLES = ("Heading", "Caption", "TOC")


@dataclass
class Display:
    start: int
    end: int
    latex: str
    remark: tuple[int, int] | None = None


def omml(latex: str, inline: bool = False):
    xml = mathml_to_omml(latex_to_mathml(latex))
    if inline:  # phân số trong dòng viết dạng a/b, không xếp chồng làm giãn dòng
        xml = xml.replace("<m:f>", '<m:f><m:fPr><m:type m:val="lin"/></m:fPr>')
    return parse_xml(xml.replace("<m:oMath>", f'<m:oMath xmlns:m="{OMML_NS}">', 1).encode("utf-8"))


def _block(text: str, spans: list[MathSpan]) -> tuple[Display | None, int]:
    """Cả đoạn là công thức: trả về công thức và vị trí bắt đầu nhãn cũ cần bỏ."""
    end = len(text.rstrip())
    label = LABEL.search(text[:end])
    if label:
        end = label.start()
    remark = REMARK.search(text[:end])
    if remark and re.search(r"[^\W\d_]{2,}", remark.group()) and not any(
            s.start < remark.start() < s.end for s in spans):
        remark_range = (remark.start() + len(remark.group()) - len(remark.group().lstrip()), remark.end())
        end = remark.start()
    else:
        remark_range = None
    core = [s for s in spans if s.end <= end]
    if not core or any(s.explicit in ("code", "inline") for s in core):
        return None, len(text)
    lead = text[:core[0].start].rstrip()
    if lead and not lead.endswith(":"):
        return None, len(text)
    latex = core[0].latex
    for left, right in pairwise(core):
        gap = " ".join(text[left.end:right.start].split())
        if gap not in CONNECTORS:
            return None, len(text)
        latex += CONNECTORS[gap] + right.latex
    tail = text[core[-1].end:end].strip()
    if tail not in ("", ".", ",", ";"):
        return None, len(text)
    if not any(s.relation or s.compound or s.explicit == "display" for s in core):
        return None, len(text)
    return Display(core[0].start, end, latex + tail, remark_range), label.start() if label else len(text)


def _inline_displays(text: str, spans: list[MathSpan]) -> list[Display]:
    """Phương trình nằm giữa câu, ngoài ngoặc của câu văn."""
    displays = []
    for span in spans:
        if span.explicit == "display" or (not span.explicit and span.relation == "=" and span.compound
                                          and _prose_depth(text, spans, span.start) == 0):
            punct = re.match(r"\s*([,.;])", text[span.end:])
            end = span.end + (punct.end() if punct else 0)
            displays.append(Display(span.start, end, span.latex + (punct[1] if punct else "")))
    return displays


def _prose_depth(text: str, spans: list[MathSpan], position: int) -> int:
    masked = list(text[:position])
    for span in spans:
        masked[span.start:span.end] = " " * (min(span.end, position) - span.start) if span.start < position else []
    prose = "".join(masked)
    return prose.count("(") - prose.count(")")


def _runs(paragraph) -> list[tuple[int, int, object]]:
    """Vị trí ký tự của từng run trực tiếp, để giữ định dạng khi dựng lại đoạn."""
    from docx.text.run import Run
    bounds, cursor = [], 0
    for r in paragraph._p.findall(qn("w:r")):
        if r.find(qn("w:drawing")) is not None or r.find(qn("w:fldChar")) is not None:
            raise ValueError(f"Đoạn có toán chứa hình/field, không dựng lại được: {paragraph.text!r}")
        value = Run(r, paragraph).text
        bounds.append((cursor, cursor + len(value), r.find(qn("w:rPr"))))
        cursor += len(value)
    if cursor != len(paragraph.text):
        raise ValueError(f"Đoạn có toán chứa hyperlink/field, không dựng lại được: {paragraph.text!r}")
    return bounds


def _text_run(value: str, rpr, code: bool = False):
    from docx.text.run import Run
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(deepcopy(rpr))
    run = Run(r, None)
    run.text = value
    if code:
        run.font.name = CODE_FONT
    return r


def _emit(target, text: str, lo: int, hi: int, spans: list[MathSpan], runs) -> None:
    """Đưa text[lo:hi] vào đoạn ``target``: chữ giữ định dạng run gốc, toán thành OMML."""
    cursor = lo
    for span in [s for s in spans if lo <= s.start and s.end <= hi] + [None]:
        stop = span.start if span else hi
        for start, end, rpr in runs:
            a, b = max(start, cursor), min(end, stop)
            if a < b:
                target.append(_text_run(text[a:b], rpr))
        if span is None:
            break
        rpr = next((rpr for start, end, rpr in runs if start <= span.start < end), None)
        target.append(_text_run(span.latex, rpr, code=True) if span.explicit == "code" else omml(span.latex, inline=True))
        cursor = span.end


def _strip(text: str, lo: int, hi: int) -> tuple[int, int]:
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


def _new_paragraph(pPr, equation: bool, width):
    from docx.text.paragraph import Paragraph
    p = OxmlElement("w:p")
    if pPr is not None:
        pPr = deepcopy(pPr)
        for numPr in pPr.findall(qn("w:numPr")):
            pPr.remove(numPr)
        p.append(pPr)
    fmt = Paragraph(p, None).paragraph_format
    fmt.first_line_indent = 0
    if equation:
        fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fmt.left_indent = fmt.right_indent = 0
        fmt.keep_together = True
        fmt.tab_stops.clear_all()
        fmt.tab_stops.add_tab_stop(width // 2, WD_TAB_ALIGNMENT.CENTER)
        fmt.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
    return p


def _chapter_heading(paragraph) -> bool | None:
    """True: chương được đánh số; False: mục không số (Kết luận, Tài liệu...)."""
    if paragraph.style.name != "Heading 1":
        return None
    numPr = paragraph._p.pPr.numPr if paragraph._p.pPr is not None else None
    return not (numPr is not None and numPr.numId is not None and numPr.numId.val == 0)


def render_math(document) -> None:
    """Chuyển mọi ký hiệu toán trong đoạn văn thành OMML; công thức khối căn giữa, đánh số (chương.thứ tự)."""
    section = document.sections[-1]
    width = section.page_width - section.left_margin - section.right_margin
    chapter, count = None, 0
    for paragraph in list(document.paragraphs):
        heading = _chapter_heading(paragraph)
        if heading is not None:
            chapter, count = ((chapter or 0) + 1 if heading else None), 0
        text = paragraph.text
        spans = find_math(text)
        if not spans:
            continue
        runs = _runs(paragraph)
        displays, cut = [], len(text)
        if chapter is None and any(s.explicit == "display" for s in spans):
            raise ValueError(f"Công thức khối nằm ngoài chương được đánh số: {text!r}")
        if chapter is not None and not paragraph.style.name.startswith(PROSE_STYLES):
            block, cut = _block(text, spans)
            displays = [block] if block else _inline_displays(text, spans)
        items, cursor = [], 0
        for display in displays:
            items += [("text", *_strip(text, cursor, display.start)), ("equation", display)]
            cursor = display.remark[1] if display.remark else display.end
        items.append(("text", *_strip(text, cursor, cut)))
        items = [item for item in items if item[0] == "equation" or item[1] < item[2]]

        p = paragraph._p
        pPr = deepcopy(p.pPr) if p.pPr is not None else None
        for child in p.findall(qn("w:r")):
            p.remove(child)
        previous = None
        for kind, *args in items:
            if previous is None:
                target = p
                if kind == "equation":
                    if p.pPr is not None:
                        p.remove(p.pPr)
                    p.insert(0, _new_paragraph(pPr, True, width).pPr)
            else:
                target = _new_paragraph(pPr, kind == "equation", width)
                previous.addnext(target)
            previous = target
            if kind == "text":
                _emit(target, text, *args, spans, runs)
                continue
            display = args[0]
            count += 1
            rpr = next((rpr for start, end, rpr in runs if start <= display.start < end), None)
            target.append(_text_run("\t", rpr))
            target.append(omml(display.latex))
            if display.remark:
                target.append(_text_run(" ", rpr))
                _emit(target, text, *display.remark, spans, runs)
            target.append(_text_run(f"\t({chapter}.{count})", rpr))
