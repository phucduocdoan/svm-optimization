import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADDITIONS = ROOT / "reports" / "source" / "report_additions.md"
FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def python_blocks(path):
    return FENCE.findall(path.read_text(encoding="utf-8"))


DOC_BLOCKS = [(path.name, code) for path in sorted((ROOT / "docs").glob("*.md")) for code in python_blocks(path)]


@pytest.mark.parametrize("code", [code for _, code in DOC_BLOCKS] + python_blocks(ADDITIONS),
                         ids=[name for name, _ in DOC_BLOCKS] + ["report_additions.md"] * len(python_blocks(ADDITIONS)))
def test_python_block_runs(code, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    exec(compile(code, "<block>", "exec"), {"__name__": "__block__"})  # noqa: S102 - chạy code minh họa của chính repo


def test_report_code_blocks_are_copied_from_docs():
    docs = {code for _, code in DOC_BLOCKS}
    assert python_blocks(ADDITIONS)
    assert [code for code in python_blocks(ADDITIONS) if code not in docs] == []
