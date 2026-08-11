"""repair.py — text-layer repairs for defects that originate INSIDE the backend output.

Two defect classes found by the 2026-07-28 adversarial census cannot be fixed by better
rendering, because the damage is already present in the VLM's `content_list`:

  R9  '??' math-variable corruption — NOW FIXED AT SOURCE; this is a REGRESSION MONITOR.
      Root cause was NOT a model limitation: pdftext's `get_chars` called
      `chr(FPDFText_GetUnicode(textpage, i))`, and that PDFium API returns a UTF-16 CODE
      UNIT, so every non-BMP character (all of U+1D400-U+1D7FF) arrived as two lone
      surrogates that were then dropped or replaced by '?' apiece. Fixed upstream by
      `_merge_surrogate_pairs` in `mineru/utils/pdf_text_tool.py`; re-extraction of an
      affected paper went from 7 '??' to 0 with the real glyphs restored.

      This repair is kept deliberately, for two narrow jobs: repairing documents extracted
      BEFORE that fix, and shouting if the fork patch is ever lost in a MinerU update. On a
      healthy pipeline `repair_qq_fixed` MUST be 0 — a non-zero count in the QA notes means
      the surrogate patch regressed, not that the repair is doing useful routine work.

  R10 U+2212 minus dropped from negative numbers in body prose, so a reported -0.47%
      becomes 0.47% and the result's SIGN INVERTS. Signs survive inside table cells; only
      prose is affected.

Both are repairable because the PDF's own text layer holds the correct characters at the
block's bbox. This module recovers them by LOCAL ANCHORING rather than global alignment:
global diff fights the (legitimate) differences between the VLM's LaTeX and the text
layer's plain math, whereas the few letters either side of a defect are identical in both.

Every repair is conservative — if the evidence is not unambiguous the text is left exactly
as it was and the block is counted as unrepaired, never guessed at.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Mathematical Alphanumeric Symbols + Greek + Letterlike: the glyph classes the VLM drops.
_MATHY = (
    (0x1D400, 0x1D7FF),   # Mathematical Alphanumeric Symbols (𝐴 𝑥 𝜃 …)
    (0x0370, 0x03FF),     # Greek
    (0x2100, 0x214F),     # Letterlike Symbols (ℝ ℓ …)
    (0x1D6A4, 0x1D6A5),   # dotless i/j
)
_MAX_GLYPHS = 6           # a run longer than this is not a variable — refuse to guess
_ANCHOR = 18              # letters/digits of context used to locate a defect in the PDF text


def _is_mathy(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _MATHY)


def _strip_markup(s: str) -> str:
    """Drop HTML tags and LaTeX so the remaining letters can anchor against the text layer."""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\$[^$]*\$", " ", s)          # inline math differs wildly between engines
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    return s


def _key(s: str) -> tuple[str, list[int]]:
    """Letters/digits only, plus the index each kept char had in `s` (for mapping back)."""
    out, idx = [], []
    for i, ch in enumerate(s):
        if ch.isalnum():
            out.append(ch.lower())
            idx.append(i)
    return "".join(out), idx


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\r", " ").replace("\xad", ""))


# Separators the anchor legitimately skips over: `_key` drops punctuation, so after
# anchoring on "left panel" the cursor can land on ": " before the glyph in ": 𝑋𝑡 = ".
_SKIP = set(" \t:,;=({[<|·—–-⁢⁡")
_MAX_SKIP = 4


def _glyph_run(src: str, at: int, want: int, backwards: bool = False) -> tuple[str, int] | None:
    """The run of math glyphs adjacent to position `at`; returns (glyphs, next_index).

    THE LOAD-BEARING INVARIANT: the VLM only ever fails on Mathematical Alphanumeric /
    Greek / Letterlike codepoints — it transcribes ASCII perfectly well. So a legitimate
    replacement is made *entirely* of those glyphs. If ordinary ASCII sits at the anchor we
    are looking at the wrong place and must refuse rather than guess.
    """
    step = -1 if backwards else 1
    i, skipped = at, 0
    while 0 <= i < len(src) and src[i] in _SKIP and skipped < _MAX_SKIP:
        i += step
        skipped += 1
    got = []
    while 0 <= i < len(src) and len(got) < max(1, min(want, _MAX_GLYPHS)):
        if not _is_mathy(src[i]):
            break
        got.append(src[i])
        i += step
    if not got:
        return None
    return ("".join(reversed(got)) if backwards else "".join(got)), i


def _anchor_pos(src: str, key: str, start: int = 0) -> int | None:
    """Index in `src` just past the first occurrence of `key` at or after `start`.

    Uniqueness is only required for a cold search (start == 0). Once a cursor is
    established, taking the first match after it is both deterministic and correct,
    because a block's text and its text layer run in the same order."""
    if len(key) < 8:
        return None
    sk, smap = _key(src)
    lo = 0
    while lo < len(smap) and smap[lo] < start:
        lo += 1
    a = sk.find(key, lo)
    if a < 0:
        return None
    if start == 0 and sk.find(key, a + 1) >= 0:   # ambiguous on a cold search
        return None
    return smap[a + len(key) - 1] + 1


def repair_qq(text: str, src: str) -> tuple[str, int, int]:
    """Replace each '??' run in `text` with the real glyph(s) from the text layer `src`.

    Returns (repaired_text, n_fixed, n_left). Anchoring is on the prose immediately before
    the defect (falling back to the prose after it); the replacement is then the adjacent
    run of math glyphs. A run is left as '??' whenever the evidence is not unambiguous, so
    unrepaired corruption stays VISIBLE rather than becoming a plausible wrong answer.
    """
    if "??" not in text or not src:
        return text, 0, 0
    src = _flat(src)
    fixed = left = 0
    out, pos = [], 0
    cursor = 0        # how far through the text layer we have consumed
    for m in re.finditer(r"(?:\?\?)+", text):
        run = len(m.group(0)) // 2
        got = None
        pre_key, _ = _key(_strip_markup(text[:m.start()]))
        # Anchor at/after the cursor. Consecutive '??' runs separated only by markup share
        # the SAME alphanumeric anchor ("left panel: <sup>??</sup>??"), so without a cursor
        # they would all resolve to the first glyph; the cursor walks them along instead.
        p = _anchor_pos(src, pre_key[-_ANCHOR:], cursor)
        if p is None:
            p = _anchor_pos(src, pre_key[-_ANCHOR:])
        got = _glyph_run(src, max(p, cursor) if p is not None else cursor, run)
        if got is None and p is None:           # fall back to anchoring on what FOLLOWS
            post_key, _ = _key(_strip_markup(text[m.end():]))
            key = post_key[:_ANCHOR]
            if len(key) >= 8:
                sk, smap = _key(src)
                a = sk.find(key)
                if a >= 0 and sk.find(key, a + 1) < 0:
                    got = _glyph_run(src, smap[a] - 1, run, backwards=True)
        out.append(text[pos:m.start()])
        if got:
            out.append(got[0])
            cursor = max(cursor, got[1])
            fixed += 1
        else:
            out.append(m.group(0))
            left += 1
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), fixed, left


_NEG = re.compile(r"[−–](\d[\d,]*\.?\d*)")


def repair_signs(text: str, src: str, require_unique: bool = False) -> tuple[str, int]:
    """Re-attach a U+2212 minus the VLM dropped from a negative number in prose.

    Anchoring on surrounding words fails here: the sign usually sits next to inline maths
    ("out-of-sample $R^{2}$ of −0.47%"), and the LaTeX that must be stripped from the target
    is exactly the text the PDF renders as plain characters, so the anchors never line up.

    So the rule is value-based and deliberately strict — a number is signed only when
      (a) EVERY occurrence of it in this block's text layer carries a minus, so the source
          is unambiguous about its sign, and
      (b) the block text has it unsigned.
    Both conditions are evaluated per block, which keeps collisions rare.
    """
    if not src or ("−" not in src and "–" not in src):
        return text, 0
    src_f = _flat(src)
    n = 0
    for val in dict.fromkeys(m.group(1) for m in _NEG.finditer(src_f)):
        v = re.escape(val)
        total = len(re.findall(r"(?<![\d.])" + v + r"(?![\d])", src_f))
        negs = len(re.findall(r"[−–]\s?" + v + r"(?![\d])", src_f))
        if not total or negs != total:        # value also appears positive -> ambiguous
            continue
        if require_unique and total != 1:
            # Page-level fallback: the block's own bbox did not contain this number, so we
            # are matching against the WHOLE page. Only act when the value occurs exactly
            # once there, otherwise it could belong to a different block on the same page.
            continue
        # sign every unsigned copy in the block (the source says they are all negative)
        out, pos, hit = [], 0, 0
        for m in re.finditer(r"(?<![\d.])" + v + r"(?![\d])", text):
            before = text[:m.start()].rstrip()
            if before.endswith(("-", "−", "–")):
                continue                      # already signed
            out.append(text[pos:m.start()])
            out.append("−")                   # immediately before the digits
            pos = m.start()
            hit += 1
        if hit:
            out.append(text[pos:])
            text = "".join(out)
            n += hit
    return text, n


def repair_content_list(content_list: list[dict[str, Any]],
                        page_text: Callable[[int, Any], str] | None,
                        full_page_text: Callable[[int], str] | None = None) -> dict[str, int]:
    """Repair a content_list IN PLACE. Returns counters for the QA record.

    `page_text(page_idx, bbox)` gives the text layer inside a block; `full_page_text(page_idx)`
    is the whole page, used for captions whose text sits outside the figure's own bbox.
    """
    st = {"qq_fixed": 0, "qq_left": 0, "signs_fixed": 0, "blocks_repaired": 0}
    if page_text is None:
        return st
    for b in content_list:
        pg, bbox = b.get("page_idx"), b.get("bbox")
        if not isinstance(pg, int):
            continue
        src: list[str] | None = None
        touched = False

        def _srcs() -> list[str]:
            """Sources to try IN ORDER: the block's own bbox first (tightest, least
            ambiguous), then the whole page. They must stay separate — concatenating them
            would make every anchor occur twice and be rejected as ambiguous. The page
            fallback matters because MinerU sometimes merges prose across regions, leaving
            the anchor outside the block's own box (~80 refused repairs came from that)."""
            nonlocal src
            if src is None:
                out = []
                try:
                    if bbox:
                        out.append(page_text(pg, bbox) or "")
                except Exception:
                    pass
                if full_page_text is not None:
                    try:
                        out.append(full_page_text(pg) or "")
                    except Exception:
                        pass
                src = [s for s in out if s]
            return src

        def _fix(s: str) -> str:
            """Run every source in turn; each pass repairs what the previous could not."""
            for source in _srcs():
                if "??" not in s:
                    break
                s, f, _ = repair_qq(s, source)
                st["qq_fixed"] += f
            return s

        for field in ("text", "code_body"):
            v = b.get(field)
            if not isinstance(v, str) or not v:
                continue
            new = _fix(v) if "??" in v else v
            # Signs use ONLY the block's own bbox text. Widening the search to the page or
            # its neighbours was tried and REVERTED: it invented four minus signs, turning
            # "Fig. 23" into "Fig. -23" and a layer shape "[2,128" into "[-2,128". A value
            # can legitimately be negative somewhere nearby while THIS occurrence is a
            # figure or section reference, and no uniqueness test can tell those apart —
            # soundness requires the evidence and the target to come from the same region.
            # Signs the bbox cannot justify are left alone and recorded in known_defects.
            srcs = _srcs()
            if srcs:
                new, sg = repair_signs(new, srcs[0])
                st["signs_fixed"] += sg
            st["qq_left"] += len(re.findall(r"(?:\?\?)+", new))
            if new != v:
                b[field] = new
                touched = True

        # Captions: the text usually lies OUTSIDE the figure's own bbox, so the page-level
        # source in _srcs() is what actually resolves these (e.g. Time-MoE's model-size
        # subscripts, 207 of them, all in image_caption).
        for field in ("image_caption", "table_caption", "image_footnote", "table_footnote"):
            v = b.get(field)
            if not isinstance(v, list):
                continue
            for i, item in enumerate(v):
                if not isinstance(item, str) or "??" not in item:
                    continue
                new = _fix(item)
                st["qq_left"] += len(re.findall(r"(?:\?\?)+", new))
                if new != item:
                    v[i] = new
                    touched = True

        if touched:
            st["blocks_repaired"] += 1
    return st
