"""
postprocess.py — Build clean, image-free Markdown from MinerU's content_list.json,
with hard guards against silent content loss.

Precision core of the local deployment. It never silently drops a table, equation,
reference, or footnote. Anything that fails recognition is FLAGGED **and its text is
salvaged** — guards are purely additive (warn + preserve), never subtractive.

Verified against MinerU 3.4.0 (`mineru/backend/vlm/vlm_middle_json_mkcontent.py:424-533`,
`enum_class.py`):
  * text-bearing types carry `text`: text, ref_text, phonetic, header, footer,
    page_number, aside_text, page_footnote
  * titles are type="text" WITH a `text_level` key (no distinct "title" type)
  * list  -> {"list_items":[...]}                    ;  code -> {"code_body": <pre-rendered>, ...}
  * equation -> {"text":"$$...$$", "text_format":"latex"}
  * table -> {"table_caption":[...], "table_footnote":[...]}; `table_body` present ONLY
    when HTML was produced, else absent (a FAILED table arrives with just `img_path`).
  * `code_body` is ALREADY RENDERED by MinerU: a fenced ```lang block (code) or a
    <div class="mineru-algorithm"> HTML block (algorithm) — emit verbatim, don't re-wrap.

OTSL failure tokens (raw OTSL leaked into table_body): <fcel> <ecel> <lcel> <ucel> <xcel> <nl>.

Policy: keep every text-bearing block incl. references (ref_text) and footnotes
(page_footnote/aside_text); for figures/charts keep the CAPTION + FOOTNOTE TEXT (an
image-blind agent's only description of the figure) while dropping the image itself;
drop page furniture (header/footer/page_number).
"""

from __future__ import annotations

import html as _html
import os
import re
from typing import Any

TEXT_TYPES: frozenset[str] = frozenset(
    {"text", "ref_text", "phonetic", "aside_text", "page_footnote"}
)
# Figures/charts: the image is dropped (useless to an image-blind agent) but its caption
# and footnote TEXT are kept — see _figure_texts / render_item.
FIGURE_TYPES: frozenset[str] = frozenset({"image", "chart"})
DROP_TYPES: frozenset[str] = frozenset(
    {"header", "footer", "page_number"}
)
OTSL_TOKENS: tuple[str, ...] = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>")
_EQ_DELIMS = ("$$", "$", r"\[", r"\]", r"\(", r"\)")
_TAG_RE = re.compile(r"<[^>]+>")


# --- guards ----------------------------------------------------------------------

def equation_body(text: str) -> str:
    """Equation content with delimiters and whitespace stripped."""
    s = (text or "").strip()
    changed = True
    while changed:
        changed = False
        for d in _EQ_DELIMS:
            if len(s) >= 2 * len(d) and s.startswith(d) and s.endswith(d):
                s = s[len(d):len(s) - len(d)].strip()
                changed = True
    return s


def equation_status(item: dict[str, Any]) -> tuple[bool, str]:
    text = item.get("text")
    if text is None:
        return False, "equation has no `text` (image-only / recognition failed)"
    body = equation_body(text)
    if body == "" or re.fullmatch(r"[\\\s{}]*", body or ""):
        return False, "equation LaTeX body is empty"
    return True, ""


def table_status(item: dict[str, Any]) -> tuple[bool, str]:
    body = item.get("table_body")
    if not body or not str(body).strip():
        if item.get("img_path"):
            return False, "table has no HTML body (image-only / recognition failed)"
        return False, "table has empty body"
    low = str(body).lower()
    if "<table" not in low or "</table>" not in low:
        return False, "table_body is not valid HTML (<table>…</table> missing)"
    hit = next((t for t in OTSL_TOKENS if t in low), None)
    if hit:
        return False, f"table_body contains raw OTSL token {hit} (otsl->html failed)"
    return True, ""


def salvage_html_text(body: Any) -> str:
    """Plain text of an HTML/OTSL fragment (never deletes cell text)."""
    txt = _html.unescape(_TAG_RE.sub(" ", str(body or "")))
    return re.sub(r"\s+", " ", txt).strip()


# Text-family blocks that the backend can hand back EMPTY. These are KEPT types, so an
# empty one is a text region the layout model found but the VLM failed to transcribe —
# i.e. exactly where body text goes missing. Historically these returned None and were
# counted as "dropped", which made the loss SILENT (the table/equation guards never see
# them). R8: salvage from the PDF text layer if we can, flag if we can't, never silent.
_EMPTIABLE: frozenset[str] = TEXT_TYPES | {"list"}

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
              "–": "-", "—": "-", "’": "'", "“": '"', "”": '"', "\x02": "", "­": ""}


def _cmp_key(s: str) -> str:
    """Comparison key immune to whitespace, case, ligatures, soft hyphens and punctuation —
    so 'already present elsewhere?' cannot be fooled by reflow or hyphenation."""
    for k, v in _LIGATURES.items():
        s = s.replace(k, v)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def salvage_is_trustworthy(text: str, pua_threshold: float = 0.05) -> bool:
    """Guard against salvaging GARBAGE. On a scanned/badly-encoded PDF the text layer is
    Private-Use-Area gibberish; injecting that would be worse than the empty block it
    replaces. Mirrors coverage.pua_ratio and its 0.05 threshold so both agree on what
    "unreliable text layer" means."""
    if not text:
        return False
    pua = sum(1 for ch in text if 0xE000 <= ord(ch) <= 0xF8FF)
    return (pua / len(text)) <= pua_threshold


def empty_kept_block(item: dict[str, Any]) -> bool:
    """True if this is a KEPT text-family block that renders to nothing."""
    itype = item.get("type", "")
    if itype not in _EMPTIABLE:
        return False
    if itype == "list":
        return not [x for x in (item.get("list_items") or []) if str(x).strip()]
    return not (item.get("text") or "").strip()


def _already_present(text: str, corpus_key: str) -> bool:
    """Is `text` already somewhere in the document? Windowed match on the normalized key.
    Most empty blocks are benign duplicates (a multi-page reference list is attributed to
    ONE page and the continuation pages carry empty placeholders) — re-emitting those
    would duplicate content, so they are dropped quietly and only counted."""
    n = _cmp_key(text)
    if not n:
        return True                      # nothing there at all
    if len(n) <= 60:
        return n in corpus_key
    step = max(40, (len(n) - 40) // 3)
    wins = [n[i:i + 40] for i in range(0, len(n) - 40, step)][:3]
    return any(w in corpus_key for w in wins)


# --- rendering helpers -----------------------------------------------------------

def _heading_level(item: dict[str, Any]) -> int:
    """Robust heading level: MinerU emits int, but tolerate digit-strings; ignore bool."""
    lvl = item.get("text_level")
    if isinstance(lvl, bool):
        return 0
    if isinstance(lvl, str) and lvl.strip().isdigit():
        lvl = int(lvl)
    return lvl if isinstance(lvl, int) and lvl > 0 else 0


def _heading(level: int, text: str) -> str:
    return f"{'#' * max(1, min(int(level), 6))} {text}".rstrip()


def _flag_comment(kind: str, page_idx: Any, reason: str) -> str:
    pg = page_idx + 1 if isinstance(page_idx, int) else page_idx   # 1-based PDF page for the reviewer
    return f"<!-- ⚠ MINERU-FLAG {kind} pdf_page={pg}: {reason} -->"


def _captions(item: dict[str, Any], key: str) -> list[str]:
    return [str(v).strip() for v in (item.get(key) or []) if str(v).strip()]


# A bare sub-panel label like "(a)" / "b." describes an invisible panel — no value to an
# image-blind agent — so it is filtered out of the kept figure text.
_PANEL_LABEL = re.compile(r"\(?[A-Za-z0-9]{1,2}\)?[.:]?")


def _figure_texts(item: dict[str, Any]) -> list[str]:
    """Caption + footnote text of a figure/chart, minus bare sub-panel labels."""
    return [t for t in (_captions(item, "image_caption") + _captions(item, "image_footnote"))
            if not _PANEL_LABEL.fullmatch(t)]


# --- per-item rendering ----------------------------------------------------------

def render_item(item: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    """Render one content_list item. `None` = intentionally dropped (figure / furniture).
    Failed tables/equations are FLAGGED and their text SALVAGED, never dropped."""
    itype = item.get("type", "")
    page_idx = item.get("page_idx")
    flags: list[dict[str, Any]] = []

    if itype in DROP_TYPES:
        return None, flags

    if itype in FIGURE_TYPES:
        # Drop the image (an image-blind agent can't use it); keep its caption/footnote text.
        return ("\n\n".join(_figure_texts(item)) or None), flags

    if itype == "text":
        text = (item.get("text") or "").strip()
        if not text:
            return None, flags
        lvl = _heading_level(item)
        return (_heading(lvl, text) if lvl else text), flags

    if itype in ("ref_text", "phonetic", "aside_text", "page_footnote"):
        return ((item.get("text") or "").strip() or None), flags

    if itype == "list":
        items = [str(x).strip() for x in (item.get("list_items") or []) if str(x).strip()]
        if not items:
            return None, flags
        return "\n".join(f"- {x}" for x in items), flags

    if itype == "equation":
        ok, reason = equation_status(item)
        if ok:
            return (item.get("text") or "").strip(), flags
        flags.append({"kind": "equation", "page_idx": page_idx,
                      "bbox": item.get("bbox"), "reason": reason})
        raw = (item.get("text") or "").strip()
        if equation_body(raw):                     # preserve any LaTeX content
            return f"{_flag_comment('EQUATION', page_idx, reason)}\n{raw}", flags
        return _flag_comment("EQUATION", page_idx, reason), flags

    if itype == "table":
        parts: list[str] = list(_captions(item, "table_caption"))
        ok, reason = table_status(item)
        if ok:
            parts.append(str(item["table_body"]).strip())
        else:
            flags.append({"kind": "table", "page_idx": page_idx,
                          "bbox": item.get("bbox"), "reason": reason})
            parts.append(_flag_comment("TABLE", page_idx, reason))
            salvaged = salvage_html_text(item.get("table_body"))
            if salvaged:                           # never delete cell text
                parts.append(salvaged)
        parts.extend(_captions(item, "table_footnote"))
        return ("\n\n".join(p for p in parts if p) or None), flags

    if itype == "code":
        body = (item.get("code_body") or "").strip()
        chunk: list[str] = list(_captions(item, "code_caption"))
        if body:
            chunk.append(body)                     # already fenced (code) / HTML (algorithm)
        elif item.get("img_path"):
            flags.append({"kind": "code", "page_idx": page_idx,
                          "bbox": item.get("bbox"), "reason": "code has no body (image-only)"})
            chunk.append(_flag_comment("CODE", page_idx, "code body missing (image-only)"))
        return ("\n\n".join(chunk) or None), flags

    # Unknown type: never drop silently — salvage any text and flag.
    salvage = item.get("text") or salvage_html_text(item.get("table_body")) or ""
    flags.append({"kind": "unknown_type", "page_idx": page_idx,
                  "bbox": item.get("bbox"), "reason": f"unrecognized block type {itype!r}"})
    marker = _flag_comment("UNKNOWN", page_idx, f"type={itype!r}")
    return ((f"{marker}\n{salvage}".strip()) or marker), flags


# --- whole-document build --------------------------------------------------------

def _recover_empty(item: dict[str, Any], salvage_bbox: Any, corpus_key: str,
                   stats: dict[str, int]) -> tuple[str | None, list[dict[str, Any]]]:
    """Deal with a KEPT text-family block that rendered to nothing.

    Three outcomes, none of them a silent drop:
      recovered — the PDF text layer has text at its bbox that is NOT already in the
                  document → emit it behind a SALVAGED flag;
      benign    — the text IS already elsewhere (placeholder for a block attributed to
                  another page) → drop quietly, counted in `empty_blocks_benign`;
      unrecovered — nothing in the text layer there (scanned page / genuinely blank
                  region) → counted in `empty_blocks_unrecovered` and surfaced as a
                  document-level review signal rather than a noisy inline flag.
    """
    page_idx, bbox = item.get("page_idx"), item.get("bbox")
    itype = item.get("type", "")
    text = ""
    if salvage_bbox is not None and bbox and isinstance(page_idx, int):
        try:
            text = (salvage_bbox(page_idx, bbox) or "").strip()
        except Exception:
            text = ""
    if text and not salvage_is_trustworthy(text):
        stats["empty_blocks_unrecovered"] += 1     # text layer is PUA gibberish — refuse it
        return None, []
    if text and not _already_present(text, corpus_key):
        stats["salvaged_blocks"] += 1
        stats["salvaged_chars"] += len(text)
        reason = f"empty {itype} block — body text recovered from the PDF text layer"
        return (f"{_flag_comment('SALVAGED', page_idx, reason)}\n{text}",
                [{"kind": "salvaged", "page_idx": page_idx, "bbox": bbox, "reason": reason}])
    if text or salvage_bbox is None:
        stats["empty_blocks_benign"] += 1
        return None, []
    stats["empty_blocks_unrecovered"] += 1
    return None, []

def build_markdown(content_list: list[dict[str, Any]], *, source_name: str = "",
                   salvage_bbox: Any = None) -> dict[str, Any]:
    """Render the content_list to Markdown.

    `salvage_bbox` is an optional callable (page_idx, bbox) -> str returning the PDF's own
    text layer inside that box. When supplied, a KEPT text-family block that came back
    empty is recovered from the text layer instead of vanishing (R8). Recovered text is
    emitted with a SALVAGED flag so the reader knows its provenance; a block whose text is
    already elsewhere in the document is dropped quietly and merely counted.
    """
    chunks: list[str] = []
    flags: list[dict[str, Any]] = []
    dropped_types: dict[str, int] = {}
    last_page: int | None = None   # emit a `<!-- page N -->` marker when kept content changes page
    stats: dict[str, int] = {
        "blocks_total": 0, "blocks_kept": 0, "blocks_dropped": 0,
        "headings": 0, "paragraphs": 0, "equations": 0, "tables": 0,
        "lists": 0, "code": 0, "figure_captions": 0, "references": 0, "footnotes": 0,
        "flagged_tables": 0, "flagged_equations": 0, "flagged_other": 0,
        "salvaged_blocks": 0, "salvaged_chars": 0, "empty_blocks_benign": 0,
        "empty_blocks_unrecovered": 0,
    }

    # Pass 1 — what will the document contain? Needed to tell a genuinely-lost empty block
    # from a benign placeholder whose text is already rendered elsewhere. render_item is
    # pure, so rendering twice is safe and cheap.
    rendered = [render_item(it) for it in content_list]
    corpus_key = _cmp_key("\n".join(m for m, _ in rendered if m))

    for idx, item in enumerate(content_list):
        stats["blocks_total"] += 1
        itype = item.get("type", "")
        md, item_flags = rendered[idx]

        if md is None and empty_kept_block(item):
            md, extra = _recover_empty(item, salvage_bbox, corpus_key, stats)
            item_flags = item_flags + extra

        flags.extend(item_flags)
        for f in item_flags:
            if f["kind"] == "salvaged":
                continue      # a REPAIR, not a defect — counted in stats['salvaged_blocks']
            key = {"table": "flagged_tables", "equation": "flagged_equations"}.get(
                f["kind"], "flagged_other")
            stats[key] += 1

        if md is None:
            stats["blocks_dropped"] += 1
            dropped_types[itype] = dropped_types.get(itype, 0) + 1
            continue

        stats["blocks_kept"] += 1
        if itype == "text":
            stats["headings" if _heading_level(item) else "paragraphs"] += 1
        elif itype == "equation":
            stats["equations"] += 1
        elif itype == "table":
            stats["tables"] += 1
        elif itype == "list":
            stats["references" if item.get("sub_type") == "ref_text" else "lists"] += 1
        elif itype == "code":
            stats["code"] += 1
        elif itype in FIGURE_TYPES:
            stats["figure_captions"] += 1
        elif itype == "ref_text":
            stats["references"] += 1
        elif itype in ("page_footnote", "aside_text"):
            stats["footnotes"] += 1

        pg = item.get("page_idx")
        if isinstance(pg, int) and pg != last_page:   # page marker only where kept content lands
            chunks.append(f"<!-- page {pg + 1} -->")   # 1-based PDF page, matches the flag comments
            last_page = pg
        chunks.append(md)

    return {
        "markdown": "\n\n".join(chunks).strip() + "\n",
        "flags": flags,
        "stats": stats,
        "dropped_types": dropped_types,
    }


# --- QA/provenance header (folds the trust signal into the .md) -------------------

def _safe_comment(s: Any) -> str:
    """Collapse any run of 2+ hyphens so dynamic text can never close (`-->`) or
    confuse (bare `--`) the surrounding HTML comment."""
    return re.sub(r"-{2,}", "-", str(s))


def build_qa_header(qa: dict[str, Any]) -> str:
    """A compact, agent-facing provenance + trust banner as ONE HTML-comment block.

    Invisible when the Markdown is rendered, but visible to an agent reading the raw
    text — it carries the qa.json signal an agent needs to know WHICH extraction to trust
    (verdict + why + coverage/cross-check recall + how many blocks are low-confidence),
    so the trust signal travels with the content instead of in a sidecar the agent may
    never open. The coverage/cross-check comparators strip HTML comments before scoring,
    so this header never affects recall."""
    st = qa.get("stats") or {}
    cov = qa.get("coverage") or {}
    xc = qa.get("crosscheck") or {}
    ocr = qa.get("ocr_enabled")
    src = os.path.basename(str(qa.get("source_pdf", ""))) or "?"
    body_mode = "VLM OCR" if ocr else ("native text" if ocr is False else "auto")

    meta = f"source: {src}"
    pages = qa.get("pages")
    if isinstance(pages, int) and pages > 0:
        meta += f" | pages: {pages}"
    # equations/tables are reliably typed by MinerU; `references` is NOT (reference sections
    # are often typed as plain text -> undercount), so it is deliberately omitted from the header.
    meta += (f" | body text: {body_mode}"
             f" | equations: {st.get('equations', 0)} | tables: {st.get('tables', 0)}")

    lines = ["<!-- MINERU-QA", _safe_comment(meta)]
    # A confirmed audit finding outranks the pipeline's own opinion: a paper whose headline
    # table lost a column must never announce itself as "clean" just because no guard fired.
    audited = qa.get("known_defects") or []
    worst = "high" if any(f.get("severity") == "high" for f in audited) else (
        "medium" if audited else None)
    if worst == "high":
        lines.append("verdict: UNRELIABLE IN PLACES — audited against the source PDF")
    elif qa.get("needs_review"):
        lines.append("verdict: NEEDS REVIEW")
        for r in qa.get("review_reasons") or []:
            lines.append(_safe_comment(f"  - {r}"))
    elif worst == "medium":
        lines.append("verdict: minor audited defects — see below")
    else:
        lines.append("verdict: clean")

    # Where an arXiv paper has an author-source reference (texref), say so immediately after
    # the verdict. An agent implementing from this file must know that a non-OCR, non-VLM
    # rendering of the same mathematics exists one directory away, and which of the two wins
    # when they disagree. Placed high in the header so a truncated read still catches it.
    if qa.get("texref"):
        # The path must survive VERBATIM — it is a filesystem key, not prose. _safe_comment
        # collapses every run of 2+ hyphens, which silently rewrote a real bundle named
        # "... Finance -- an Application ..." to "... Finance - an Application ..." and left
        # the pointer resolving to nothing. Only the literal `-->` can close the comment, so
        # the prose is sanitised as usual and the path is spliced in afterwards through a
        # hyphen-free placeholder, guarded against that one sequence.
        path = str(qa["texref"]).replace("-->", "--&gt;")
        lines.append(_safe_comment(
            "AUTHOR SOURCE: verbatim mathematics from this paper's own LaTeX is in "
            "\x00TEXREF\x00/ — equations.md (display equations), inline.md (inline maths + "
            "notation inventory), paper.flat.tex (full flattened source). It is derived "
            "from the source, not from the rendered page, so it carries no OCR or "
            "layout-inference risk. WHERE THIS FILE AND THAT REFERENCE DISAGREE ABOUT "
            "MATHEMATICS, THE REFERENCE IS AUTHORITATIVE.").replace("\x00TEXREF\x00", path))

    trust: list[str] = []
    if "word_recall" in cov:
        trust.append(f"word-recall {cov['word_recall']}")
    if "numeral_recall" in cov:
        trust.append(f"numeral-recall {cov['numeral_recall']} (figure/table-confounded)")
    if xc and "pipeline_vs_hybrid_numeral_recall" in xc:
        trust.append(f"cross-check {xc['pipeline_vs_hybrid_numeral_recall']}")
    if trust:
        lines.append(_safe_comment("coverage: " + " | ".join(trust)))

    nflag = (st.get("flagged_tables", 0) + st.get("flagged_equations", 0)
             + st.get("flagged_other", 0))
    if nflag:
        lines.append(f"{nflag} low-confidence block(s) marked inline below with MINERU-FLAG.")
    # Repairs are provenance, not defects — an agent must know some prose came from the
    # PDF text layer rather than the VLM, so it travels in the header too (R8).
    if st.get("salvaged_blocks"):
        lines.append(f"{st['salvaged_blocks']} empty text block(s) ({st.get('salvaged_chars', 0)} chars) "
                     f"were RECOVERED from the PDF text layer — marked MINERU-FLAG SALVAGED inline.")

    # Defects an adversarial PDF-vs-Markdown audit CONFIRMED against the source. These are
    # things this pipeline cannot repair (a table column the recognizer dropped, an altered
    # digit); the honest thing is to make the agent aware rather than fabricate a fix, so
    # the finding travels inside the file an agent actually reads.
    kd = qa.get("known_defects") or []
    if kd:
        hi = sum(1 for f in kd if f.get("severity") == "high")
        lines.append(_safe_comment(
            f"AUDITED: {len(kd)} verified extraction defect(s) in this paper ({hi} high). "
            f"DO NOT quote the affected values without checking the source PDF:"))
        for f in kd[:12]:
            pg = f.get("page")
            lines.append(_safe_comment(
                f"  [{f.get('severity', '?')}] {('p' + str(pg)) if pg else 'see text'}: {f.get('summary', '')}"))
        if len(kd) > 12:
            lines.append(f"  … and {len(kd) - 12} more (full list in known_defects.json)")
    lines.append("-->")
    return "\n".join(lines)


# --- plain-text extraction (for coverage / cross-check) --------------------------

def item_plain_text(item: dict[str, Any]) -> str:
    itype = item.get("type", "")
    if itype in DROP_TYPES or itype == "equation":
        return ""
    if itype in FIGURE_TYPES:
        return " ".join(_figure_texts(item))
    if itype in TEXT_TYPES or itype == "text":
        return item.get("text") or ""
    if itype == "list":
        return " ".join(str(x) for x in (item.get("list_items") or []))
    if itype == "table":
        pieces = _captions(item, "table_caption") + _captions(item, "table_footnote")
        if item.get("table_body"):
            pieces.append(salvage_html_text(item.get("table_body")))
        return " ".join(pieces)
    if itype == "code":
        return (item.get("code_body") or "") + " " + " ".join(_captions(item, "code_caption"))
    return item.get("text") or ""


def document_plain_text(content_list: list[dict[str, Any]]) -> str:
    return " ".join(t for t in (item_plain_text(i) for i in content_list) if t)


# --- self-test (runnable without models) -----------------------------------------

if __name__ == "__main__":
    sample = [
        {"type": "text", "text": "1 Introduction", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "The value is 3.14159 and n=42.", "page_idx": 0},
        {"type": "equation", "text": "$$E = mc^2$$", "text_format": "latex", "page_idx": 0},
        {"type": "equation", "text": "$$\n\n$$", "text_format": "latex", "page_idx": 1},
        {"type": "equation", "img_path": "x.jpg", "page_idx": 1},
        {"type": "table", "table_body": "<table><tr><td>a</td></tr></table>",
         "table_caption": ["Table 1"], "page_idx": 2},
        {"type": "table", "img_path": "t.jpg", "table_caption": ["Table 2 lost"], "page_idx": 2},
        {"type": "table", "table_body": "<tr><td>DATA 42 alpha</td></tr>", "page_idx": 2},  # rows-only -> flag+salvage
        {"type": "ref_text", "text": "[1] Author, Title, 2020.", "page_idx": 3},
        {"type": "page_footnote", "text": "* significant at 5%", "page_idx": 3},
        {"type": "list", "sub_type": "ref_text", "list_items": ["Ref A 1999", "Ref B 2001"], "page_idx": 3},
        {"type": "image", "img_path": "fig.jpg",
         "image_caption": ["Figure 1: A plot of X vs Y."], "image_footnote": ["Source: authors."],
         "page_idx": 4},
        {"type": "chart", "img_path": "c.jpg", "image_caption": ["(a)"], "page_idx": 4},  # bare label -> dropped
        {"type": "header", "text": "Journal of X", "page_idx": 4},
        {"type": "code", "code_body": "```python\nfor i in range(n):\n    x += 1\n```", "page_idx": 5},
    ]
    out = build_markdown(sample, source_name="selftest")
    md = out["markdown"]
    print(md)
    print("STATS:", out["stats"])
    print("DROPPED_TYPES:", out["dropped_types"])

    assert out["stats"]["flagged_tables"] == 2, out["stats"]
    assert out["stats"]["flagged_equations"] == 2, out["stats"]
    assert "significant at 5%" in md, "footnote dropped!"
    assert "[1] Author" in md, "reference dropped!"
    assert "Table 2 lost" in md, "failed-table caption dropped!"
    assert "DATA 42 alpha" in md, "failed-table cell text DELETED — content loss!"  # salvage
    assert "Ref A 1999" in md and "Ref B 2001" in md, "reference list dropped!"
    assert "Journal of X" not in md, "header should be dropped"
    assert "fig.jpg" not in md and "c.jpg" not in md, "image path should be dropped"
    assert "Figure 1: A plot of X vs Y." in md, "figure CAPTION text must be kept"
    assert "Source: authors." in md, "figure footnote text must be kept"
    assert "(a)" not in md, "bare sub-panel label should be filtered out"
    assert out["stats"]["figure_captions"] == 1, out["stats"]  # image kept; chart '(a)' dropped
    assert md.count("```") == 2, "code block should be verbatim (single fence), not re-wrapped"
    assert "E = mc^2" in md
    assert out["stats"]["references"] == 2, out["stats"]  # ref_text block + ref_text list

    # page markers: one per page that has KEPT content, 1-based, in reading order
    assert "<!-- page 1 -->" in md, "missing page-1 marker"
    assert "<!-- page 3 -->" in md, "missing page-3 marker (tables page)"
    assert "<!-- page 6 -->" in md, "missing page-6 marker (code, page_idx=5)"
    assert "<!-- page 5 -->" in md, "page 5 (idx 4) now carries the kept figure caption"

    # QA/provenance header folds the qa.json trust signal into the .md as ONE valid comment
    qa_sample = {
        "source_pdf": "/papers/Informed_Trading.pdf", "pages": 23, "needs_review": True,
        "review_reasons": ["1 table(s) failed recognition (flagged in .md)",
                           "cross-check: pipeline has >=2-digit numbers absent from hybrid"],
        "stats": {"equations": 14, "tables": 6, "references": 40,
                  "flagged_tables": 1, "flagged_equations": 0, "flagged_other": 0},
        "coverage": {"word_recall": 0.98, "numeral_recall": 0.91},
        "crosscheck": {"pipeline_vs_hybrid_numeral_recall": 0.99},
        "ocr_enabled": True,
    }
    hdr = build_qa_header(qa_sample)
    print("\nHEADER:\n" + hdr)
    assert hdr.startswith("<!-- MINERU-QA") and hdr.rstrip().endswith("-->")
    assert hdr.count("-->") == 1, "dynamic text must not inject a comment terminator (>= reason)"
    assert "source: Informed_Trading.pdf" in hdr and "pages: 23" in hdr
    assert "NEEDS REVIEW" in hdr and "word-recall 0.98" in hdr and "cross-check 0.99" in hdr
    assert "1 low-confidence block(s) marked inline" in hdr
    hdr_clean = build_qa_header({"source_pdf": "x/Clean.pdf", "pages": 5, "needs_review": False,
                                 "stats": {"equations": 0, "tables": 0, "references": 3},
                                 "coverage": {"word_recall": 1.0, "numeral_recall": 0.99},
                                 "crosscheck": {"pipeline_vs_hybrid_numeral_recall": 1.0},
                                 "ocr_enabled": False})
    assert "verdict: clean" in hdr_clean and "native text" in hdr_clean
    print("\nAll self-tests passed.")
