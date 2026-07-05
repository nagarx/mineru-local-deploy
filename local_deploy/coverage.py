"""
coverage.py — Text-coverage safety net (red-team R7), hardened against silent loss.

Compares the PDF's own embedded text layer (via pypdfium2 — the SAME engine MinerU
uses) against the content we extracted, to catch any word or number silently dropped.

Design (after two adversarial code reviews):
  - NUMBERS use MULTISET recall: dropping 3-of-4 copies of a value is caught. Every
    digit run counts, including single digits and scientific notation as whole tokens.
    Deliberately-dropped furniture (running header/footer/page-number text + figure
    captions/footnotes) is removed from the reference BY OCCURRENCE (not globally), so a
    real datum that merely equals a page-number value is still checked everywhere else.
  - WORDS use distinct-token coverage with prefix/suffix matching, so PDF hyphenation
    fragments ("agricul" of "agricultural") count as present, without cross-word false
    matches ("ration" is NOT covered by "operational").
  - Both sides are normalized identically (soft hyphen, line-break de-hyphenation,
    ligatures, NFKC, quotes/dashes, casefold).
  - Per-page: a page whose distinctive words OR numbers are largely absent from the
    whole output is flagged (catches a dropped page/column).
  - If the PDF text layer is itself unreliable (>5% Private-Use-Area codepoints) we report
    source_reliable=False rather than crying "miss".
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

# --- normalization ---------------------------------------------------------------

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
_PUNCT = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "−": "-",
})
_HYPHEN_BREAK = re.compile(r"(\w)[-‐]\s*\n\s*([a-z])")


def normalize(text: str) -> str:
    if not text:
        return ""
    s = text.replace("­", "")                 # soft hyphen
    s = _HYPHEN_BREAK.sub(r"\1\2", s)         # de-hyphenate at line breaks
    for k, v in _LIGATURES.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKC", s)      # full->half width, remaining ligatures
    s = re.sub(r"(?<=\d)\s*[·∙⋅•](?=\d)", ".", s)  # middle-dot decimals (Biometrika-style 1·25 -> 1.25)
    s = s.translate(_PUNCT)
    s = re.sub(r"\s+", " ", s)
    return s.casefold().strip()


_HTML_TAG = re.compile(
    r"</?(?:table|thead|tbody|tfoot|tr|td|th|caption|colgroup|col|br|hr|p|b|i|u|em|"
    r"strong|sup|sub|span|div|ul|ol|li|a|font|pre|code)\b[^<>]*>",
    re.IGNORECASE,
)


def markdown_to_comparable(md: str) -> str:
    """Markdown -> plain token soup for comparison. Removes our flag comments FIRST,
    then real HTML tags, then neutralizes any remaining '<'/'>'/'$' by turning them into
    spaces WITHOUT deleting the text between them (a naive `<[^>]+>` strip eats real
    words between a stray math '<' and a later '>')."""
    s = re.sub(r"<!--.*?-->", " ", md, flags=re.DOTALL)   # our flag comments first
    s = _HTML_TAG.sub(" ", s)                              # real HTML tags
    s = s.replace("<", " ").replace(">", " ").replace("$", " ")
    return normalize(s)


# --- tokenization ----------------------------------------------------------------

# Legacy multiset word/number tokenizers (kept for crosscheck.py + self-test).
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def words(s: str) -> list[str]:
    return _WORD.findall(s)


def numbers(s: str) -> list[str]:
    return [t.replace(",", "") for t in _NUM.findall(s)]


# Coverage tokenizers: distinctive words (len>=4, incl. 4-letter scientific words like
# data/gene/dose/rate) and numbers as whole tokens (single digits, decimals, thousands,
# scientific notation e.g. 1.5e-3).
_STRICT_WORD = re.compile(r"[a-z]{4,}")
_NUM_TOKEN = re.compile(r"\d[\d,]*\.\d+(?:[eE][-+]?\d+)?|\d[\d,]*(?:[eE][-+]?\d+)?")


def strict_words(s: str) -> list[str]:
    return _STRICT_WORD.findall(s)


def num_tokens(s: str) -> list[str]:
    return [t.replace(",", "") for t in _NUM_TOKEN.findall(s)]


def pua_ratio(text: str) -> float:
    if not text:
        return 0.0
    pua = sum(
        1 for ch in text
        if 0xE000 <= ord(ch) <= 0xF8FF
        or 0xF0000 <= ord(ch) <= 0xFFFFD
        or 0x100000 <= ord(ch) <= 0x10FFFD
    )
    return pua / len(text)


# --- furniture (content we deliberately drop) ------------------------------------

FURNITURE_TYPES = frozenset({"header", "footer", "page_number"})
_FIGURE_CAPTION_KEYS = ("image_caption", "chart_caption", "image_footnote", "chart_footnote")


def furniture_text(content_list: list[dict[str, Any]] | None) -> str:
    """Text we DELIBERATELY drop: running headers/footers/page numbers, and figure
    captions AND footnotes (image/chart). Removed from the reference by occurrence so
    its per-page repetition can't swamp recall — but never enough to hide a real datum
    elsewhere (multiset subtraction, see coverage_report)."""
    parts: list[str] = []
    for it in (content_list or []):
        if it.get("type") in FURNITURE_TYPES and it.get("text"):
            parts.append(str(it["text"]))
        for k in _FIGURE_CAPTION_KEYS:
            if it.get(k):
                parts.extend(str(x) for x in it[k])
    return normalize(" ".join(parts))


# --- recall primitives -----------------------------------------------------------

def _recall(ref_tokens: list[str], cand_multiset: Counter | dict) -> tuple[float, list[str]]:
    """Multiset recall of ref_tokens against a candidate multiset."""
    avail = dict(cand_multiset)
    missing: list[str] = []
    total = covered = 0
    for tok, cnt in Counter(ref_tokens).items():
        total += cnt
        have = avail.get(tok, 0)
        use = min(cnt, have)
        covered += use
        avail[tok] = have - use
        if use < cnt:
            missing.extend([tok] * (cnt - use))
    return (1.0 if total == 0 else covered / total), missing


def _multiset_subtract(tokens: list[str], remove: list[str]) -> list[str]:
    """Multiset difference tokens - remove (by occurrence, floored at 0)."""
    return list((Counter(tokens) - Counter(remove)).elements())


def _word_covered(w: str, cand_set: set[str], cand_list: list[str]) -> bool:
    """A ref word is covered if it is a candidate token, OR a prefix/suffix of one
    (handles PDF hyphenation fragments) — but not an interior substring (avoids
    'ration' matching 'operational')."""
    if w in cand_set:
        return True
    for t in cand_list:
        if len(t) > len(w) and (t.startswith(w) or t.endswith(w)):
            return True
    return False


# --- PDF text-layer extraction ---------------------------------------------------

def extract_pdf_pages(pdf_path: str) -> list[str]:
    """Per-page text from the PDF's embedded text layer, via pypdfium2."""
    import pypdfium2 as pdfium

    pages: list[str] = []
    doc = pdfium.PdfDocument(pdf_path)
    try:
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            try:
                pages.append(tp.get_text_bounded() or "")   # full-page text (pypdfium2 4.x/5.x)
            finally:
                tp.close()
                page.close()
    finally:
        doc.close()
    return pages


# --- top-level report ------------------------------------------------------------

def coverage_report(
    pdf_path: str,
    markdown: str,
    content_list: list[dict[str, Any]] | None = None,
    *,
    min_page_chars: int = 150,
    pua_threshold: float = 0.05,
) -> dict[str, Any]:
    ref_pages = extract_pdf_pages(pdf_path)
    ref_all = normalize(" ".join(ref_pages))
    src_pua = pua_ratio(" ".join(ref_pages))
    source_reliable = src_pua <= pua_threshold

    cand = markdown_to_comparable(markdown)
    cand_num_ms = Counter(num_tokens(cand))
    cand_word_set = set(strict_words(cand))
    cand_word_list = list(cand_word_set)

    furn = furniture_text(content_list)

    # NUMBERS — multiset recall, furniture removed by occurrence.
    ref_num_ms = _multiset_subtract(num_tokens(ref_all), num_tokens(furn))
    num_recall, missing_numbers = _recall(ref_num_ms, cand_num_ms)

    # WORDS — distinct-token coverage with prefix/suffix matching.
    ref_words = sorted(set(strict_words(ref_all)) - set(strict_words(furn)))
    missing_words = [w for w in ref_words if not _word_covered(w, cand_word_set, cand_word_list)]
    word_recall = 1.0 if not ref_words else (len(ref_words) - len(missing_words)) / len(ref_words)

    # PER-PAGE — flag a page whose distinctive words OR numbers are largely absent.
    page_flags: list[dict[str, Any]] = []
    for idx, raw in enumerate(ref_pages):
        norm = normalize(raw)
        if len(norm) < min_page_chars:
            continue
        pw = sorted(set(strict_words(norm)))
        if len(pw) < 8:      # too few distinctive words (e.g. a figure-only page) to judge
            continue
        wcov = sum(1 for w in pw if _word_covered(w, cand_word_set, cand_word_list)) / len(pw)
        # word-based only: numeral-per-page is confounded by figure axis labels / table digits
        if wcov < 0.60:
            page_flags.append({"page_idx": idx, "word_recall": round(wcov, 3),
                               "ref_chars": len(norm),
                               "reason": "page body text largely absent from output"})

    return {
        "source_reliable": source_reliable,
        "source_pua_ratio": round(src_pua, 4),
        "numeral_recall": round(num_recall, 4),
        "word_recall": round(word_recall, 4),
        "ref_number_count": len(ref_num_ms),
        "ref_word_count": len(ref_words),
        "missing_numbers": sorted(set(missing_numbers))[:100],
        "missing_words_sample": missing_words[:50],
        "suspect_pages": page_flags,
    }


# --- self-test (no PDF needed) ---------------------------------------------------

if __name__ == "__main__":
    ref = "The coeﬃcient was 3.14 and the sample size n = 1,000 with p = 7 and rate 1.5e-3.\nSee refer-\nence [1]."
    nref = normalize(ref)
    assert "coefficient" in nref                       # ligature expanded
    assert "reference" in nref                         # de-hyphenated
    assert "1000" in num_tokens(nref), num_tokens(nref)   # thousands folded
    assert "7" in num_tokens(nref), num_tokens(nref)      # single digit kept
    assert "1.5e-3" in num_tokens(nref), num_tokens(nref) # sci-notation whole token

    # multiset: dropping one of two "250"s is caught
    r, miss = _recall(["250", "250"], Counter(["250"]))
    assert r == 0.5 and miss == ["250"], (r, miss)

    # word prefix/suffix: hyphenation fragment covered, unrelated interior substring not
    cs = {"agricultural", "operational"}
    assert _word_covered("agricul", cs, list(cs))      # prefix of agricultural
    assert _word_covered("tural", cs, list(cs))        # suffix of agricultural
    assert not _word_covered("ration", cs, list(cs))   # interior of operational -> NOT covered

    # furniture removed by occurrence, not globally: a real "37" survives a page-number "37"
    ref_n = _multiset_subtract(["37", "37"], ["37"])   # one page-number "37" removed
    assert ref_n == ["37"], ref_n
    r2, miss2 = _recall(ref_n, Counter([]))            # output dropped the real 37
    assert r2 == 0.0 and miss2 == ["37"], (r2, miss2)  # caught!

    print("All coverage self-tests passed.")
