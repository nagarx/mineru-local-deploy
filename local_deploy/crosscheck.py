"""
crosscheck.py — Dual-backend cross-check (red-team R4/R6).

The hybrid backend can silently drop native text that isn't enclosed in a detected
layout block (it never calls `remaining_spans()`), and tables/equations are always
recognizer output. The deterministic `pipeline` backend is a second, independent
opinion. Here we flag:
  - words/numbers the pipeline extracted that are ABSENT from the hybrid output
    (possible hybrid drop — the strongest signal for R4), and
  - large divergence in table / equation counts between the two backends.

Backend-agnostic: it extracts plain text from either backend's content_list by
field name, so it does not depend on the (different) type taxonomies.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from coverage import (normalize, num_tokens, strict_words, _word_covered, _recall,
                      markdown_to_comparable)

_TAG = re.compile(r"<[^>]+>")
_TEXT_KEYS = ("text",)
_LIST_KEYS = ("list_items",)
_HTML_KEYS = ("table_body", "code_body")
_CAP_KEYS = ("table_caption", "table_footnote", "code_caption")


def content_list_plain_text(content_list: list[dict[str, Any]]) -> str:
    """Backend-agnostic plain-text extraction from a content_list (hybrid OR pipeline).
    Excludes figure blocks; strips HTML tags from tables."""
    parts: list[str] = []
    for it in content_list:
        itype = it.get("type", "")
        # skip figures and page furniture (both dropped from our Markdown), so the
        # cross-check doesn't false-flag headers/page-numbers as "pipeline-only".
        if itype in ("image", "chart", "header", "footer", "page_number"):
            continue
        for k in _TEXT_KEYS:
            if it.get(k):
                parts.append(str(it[k]))
        for k in _LIST_KEYS:
            if it.get(k):
                parts.extend(str(x) for x in it[k])
        for k in _HTML_KEYS:
            if it.get(k):
                parts.append(_TAG.sub(" ", str(it[k])))
        for k in _CAP_KEYS:
            if it.get(k):
                parts.extend(str(x) for x in it[k])
    return " ".join(parts)


def _count_types(content_list: list[dict[str, Any]]) -> Counter:
    return Counter(it.get("type", "") for it in content_list)


_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh][^>]*>", re.I)
_COLSPAN_RE = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.I)


def _table_widths(content_list: list[dict[str, Any]]) -> list[int]:
    """Modal column width of each table, page-ordered.

    Width counts colspans, so a header cell spanning two columns counts as two — that is
    what makes a DROPPED COLUMN visible as a width change rather than hiding inside a
    ragged row."""
    out: list[int] = []
    for it in content_list:
        if it.get("type") != "table":
            continue
        body = str(it.get("table_body") or "")
        widths = []
        for row in _ROW_RE.findall(body):
            w = 0
            for cell in _CELL_RE.findall(row):
                m = _COLSPAN_RE.search(cell)
                w += int(m.group(1)) if m else 1
            if w:
                widths.append(w)
        if widths:
            out.append(Counter(widths).most_common(1)[0][0])
    return out


def table_shape_divergence(hybrid_cl, pipeline_cl) -> list[dict[str, Any]]:
    """Tables where the two backends disagree on how many COLUMNS the table has (R11).

    A recognizer that silently drops a whole column — VisionTS lost the entire `Informer`
    column of its headline results table, 15 values — produces output that is perfectly
    well-formed HTML and passes every existing guard. The deterministic pipeline backend
    is an independent opinion on the column count, so a disagreement is the cheapest
    reliable signal that one of them lost data. Only compared when both backends found the
    same NUMBER of tables, otherwise the tables cannot be paired up positionally.
    """
    hy, pi = _table_widths(hybrid_cl), _table_widths(pipeline_cl)
    if not hy or len(hy) != len(pi):
        return []
    return [{"table_index": i, "hybrid_cols": a, "pipeline_cols": b}
            for i, (a, b) in enumerate(zip(hy, pi)) if a != b]


def compare(
    hybrid_cl: list[dict[str, Any]],
    pipeline_cl: list[dict[str, Any]],
    hybrid_markdown: str,
) -> dict[str, Any]:
    """Return a cross-check report. Low pipeline→hybrid recall means the hybrid
    output is missing content the deterministic pipeline found."""
    pipe_norm = normalize(content_list_plain_text(pipeline_cl))
    cand = markdown_to_comparable(hybrid_markdown)
    def _multi(s):  # >=2-digit numbers only; single digits are cross-backend noise
        return [t for t in num_tokens(s) if len(t) >= 2]

    cand_num_ms = Counter(_multi(cand))
    cand_word_set = set(strict_words(cand))
    cand_word_list = list(cand_word_set)

    # numbers pipeline found but hybrid lacks (multiset)
    nr, miss_n = _recall(_multi(pipe_norm), cand_num_ms)
    # words — prefix/suffix containment (tolerant of hyphenation + LaTeX-command notation)
    pipe_words = sorted(set(strict_words(pipe_norm)))
    miss_w = [w for w in pipe_words if not _word_covered(w, cand_word_set, cand_word_list)]
    wr = 1.0 if not pipe_words else (len(pipe_words) - len(miss_w)) / len(pipe_words)

    hy_types = _count_types(hybrid_cl)
    pi_types = _count_types(pipeline_cl)

    return {
        "pipeline_vs_hybrid_numeral_recall": round(nr, 4),
        "pipeline_vs_hybrid_word_recall": round(wr, 4),
        "numbers_in_pipeline_not_hybrid": sorted(set(miss_n))[:100],
        "words_in_pipeline_not_hybrid": sorted(set(miss_w))[:50],
        "hybrid_table_count": hy_types.get("table", 0),
        "pipeline_table_count": pi_types.get("table", 0),
        "hybrid_equation_count": hy_types.get("equation", 0),
        "pipeline_equation_count": pi_types.get("equation", 0),
        "table_shape_divergence": table_shape_divergence(hybrid_cl, pipeline_cl),
    }


if __name__ == "__main__":
    hy = [
        {"type": "text", "text": "alpha beta gamma 100 200"},
        {"type": "table", "table_body": "<table><tr><td>x 7</td></tr></table>"},
    ]
    pi = [
        {"type": "text", "text": "alpha beta gamma delta 100 200 300"},  # extra: delta, 300
        {"type": "table", "table_body": "<table><tr><td>x 7</td></tr></table>"},
    ]
    md = "alpha beta gamma 100 200\n\n<table><tr><td>x 7</td></tr></table>"
    rep = compare(hy, pi, md)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    assert "300" in rep["numbers_in_pipeline_not_hybrid"], rep
    assert "delta" in rep["words_in_pipeline_not_hybrid"], rep
    print("crosscheck self-test passed.")
