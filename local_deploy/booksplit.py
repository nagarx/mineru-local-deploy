#!/usr/bin/env python3
"""
booksplit.py — split a large book PDF into chapter-sized, timeout-safe chunk PDFs.

Why: MinerU's per-document API timeout (~60 min) makes whole books (300-500pp) fail
deterministically in the hybrid (VLM) backend. Splitting a book into ~35-page chunks
turns it into a series of paper-sized jobs, each finishing far under the timeout, each a
durable ~15-min unit, each producing an agent-loadable per-chapter Markdown file.

Strategy (robust to ANY book, with or without a table of contents):
  * If the PDF has a usable chapter outline — ONE FILE PER CHAPTER: pick the outline level
    that represents chapters, cut at each chapter start, keep the pre-chapter pages as a
    Front Matter file, sub-split any single chapter longer than MAX_PP, and coalesce only
    tiny fragments (cover/copyright/index) — two full chapters are NEVER put in one file.
  * If it has NO usable outline — fall back to fixed-size ~MAX_PP chunks (page-aligned).
Both paths are pure PAGE-RANGE EXTRACTION (pages copied, never re-rendered), so the split
is lossless: the union of the chunks is exactly the original book, byte-for-byte per page.

Usage (library):   booksplit.plan(pdf) -> chunk plan;  booksplit.split(pdf, out_dir, name)
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pypdfium2 as pdfium

TARGET_PP = 35    # (fixed-size fallback only) aim for ~chapter-sized chunks
MIN_PP = 15       # (fixed-size fallback only) a trailing chunk smaller than this merges back
MAX_PP = 55       # hard ceiling — ~13 min of hybrid, ~4x under MinerU's ~250pp/60min wall

# chapter-atomic mode: one file per chapter, cut at the outline's "chapter" level.
CHAPTER_TARGET = 28      # a chapter is ~this many pages; the level whose one-file-per-entry
                         # chunking has a median file size closest to this = the chapter level
MIN_ATOM = 6             # a chunk smaller than this is a fragment (cover/copyright/index/part-intro)
                         # that coalesces with adjacent fragments; two chunks >= MIN_ATOM never merge


def _slug(s: str, n: int = 48) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", (s or "").strip()).strip("_")[:n] or "section"


def _outline_by_level(doc) -> dict[int, list[tuple[int, str]]]:
    """Resolvable bookmark starts grouped by outline level: {level: [(page_idx, title), ...]}."""
    levels: dict[int, list[tuple[int, str]]] = {}
    try:
        for it in doc.get_toc():
            pi = it.page_index
            if pi is None:
                continue
            levels.setdefault(it.level, []).append((pi, it.title or ""))
    except Exception:
        pass
    for lvl in levels:
        levels[lvl].sort()
    return levels


def _chunk_ranges(boundaries: set[int], n_pages: int) -> list[tuple[int, int]]:
    """Accumulate the atomic segments between boundaries into ~TARGET_PP chunks, never
    exceeding MAX_PP; split any oversized boundary-free gap; merge a tiny final chunk."""
    B = sorted(set(boundaries) | {0, n_pages})
    segs = list(zip(B, B[1:]))                     # atomic segments [start,end)
    chunks: list[list[int]] = []
    cs = ce = None
    for s, e in segs:
        if cs is None:
            cs, ce = s, e
        elif (e - cs) <= MAX_PP and (ce - cs) < TARGET_PP:
            ce = e                                 # keep accumulating toward TARGET
        else:
            chunks.append([cs, ce]); cs, ce = s, e
    if cs is not None:
        chunks.append([cs, ce])
    # split any single chunk still over MAX_PP (a huge gap with no interior boundary)
    split: list[list[int]] = []
    for s, e in chunks:
        if e - s <= MAX_PP:
            split.append([s, e])
        else:
            k = s
            while k < e:
                nk = min(k + MAX_PP, e)
                split.append([k, nk]); k = nk
    # merge a too-small trailing chunk into its predecessor — but never past MAX_PP
    if (len(split) >= 2 and (split[-1][1] - split[-1][0]) < MIN_PP
            and (split[-1][1] - split[-2][0]) <= MAX_PP):
        split[-2][1] = split[-1][1]; split.pop()
    return [(s, e) for s, e in split]


def _pick_chapter_level(levels: dict[int, list], n_pages: int):
    """Pick the outline level whose one-file-per-entry chunking has a median file size closest
    to CHAPTER_TARGET — selecting 'chapters' over fine subsections (files too small) and coarse
    parts (files too large). Judged on the ACTUAL chunking (front matter merged, oversized
    chapters sub-split), not raw bookmark gaps — a level whose raw gaps look small because of
    front matter can still be the chapter level. Returns (level, entries) or None (<3 entries)."""
    import statistics
    best = None
    for lvl, ents in levels.items():
        if len([p for p, _ in ents if 0 <= p < n_pages]) < 3:
            continue
        sizes = [e - s for s, e, _ in _chapter_atomic_chunks(ents, n_pages)]
        if len(sizes) < 2:
            continue
        score = abs(statistics.median(sizes) - CHAPTER_TARGET)
        if best is None or score < best[0]:
            best = (score, lvl, ents)
    return None if best is None else (best[1], best[2])


def _merge_fragments(chunks: list) -> list:
    """Coalesce consecutive fragment chunks (each < MIN_ATOM) into one file; any chunk
    >= MIN_ATOM (a real chapter) stands alone — so tiny cover/copyright/index sections group
    into a single file while no two full chapters are ever merged."""
    out: list[list] = []                 # items: [start, end, title, all_fragments_so_far?]
    for a, b, t in chunks:
        frag = (b - a) < MIN_ATOM
        if out and out[-1][3] and frag:
            out[-1][1] = b               # extend the running fragment group
        else:
            out.append([a, b, t, frag])
    return [(a, b, t) for a, b, t, _ in out]


def _chapter_atomic_chunks(ents: list, n_pages: int) -> list:
    """One chunk per chapter: cut at each chapter-level start; pre-first-chapter pages are a
    Front Matter chunk; fragments coalesce; a chapter longer than MAX_PP is sub-split."""
    starts = sorted(p for p, _ in ents if 0 <= p < n_pages)
    title_at = {p: t for p, t in ents}
    bounds = sorted(set([0] + starts + [n_pages]))
    raw = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        title = title_at.get(a) or ("Front_Matter" if a == 0 else f"pages_{a+1}")
        raw.append((a, b, title))
    out = []
    for a, b, t in _merge_fragments(raw):
        if b - a <= MAX_PP:
            out.append((a, b, t))
        else:                            # one chapter longer than the ceiling -> even sub-split
            span = b - a
            nparts = -(-span // MAX_PP)          # ceil: fewest parts that keep each <= MAX_PP
            size = -(-span // nparts)            # ceil: even part size, so no wasteful 1-page tail
            k, p = 1, a
            while p < b:
                pe = min(p + size, b)
                out.append((p, pe, f"{t} (part {k})"))
                p, k = pe, k + 1
    return out


def plan(pdf_path: str | Path) -> dict:
    """Compute the chunk plan without writing. One file per chapter when the PDF has a usable
    chapter outline; otherwise fixed-size page chunks. Returns method + [(start,end,title)]."""
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        n = len(doc)
        pick = _pick_chapter_level(_outline_by_level(doc), n)
        if pick is None:
            chunks = [(s, e, f"pages_{s+1}_{e}") for s, e in _chunk_ranges(set(), n)]
            method = "fixed-size (no usable chapter outline)"
        else:
            lvl, ents = pick
            chunks = _chapter_atomic_chunks(ents, n)
            method = f"chapter-atomic (outline L{lvl}, {len(ents)} chapter marks)"
        return {"pages": n, "method": method, "chunks": chunks}
    finally:
        doc.close()


def split(pdf_path: str | Path, out_dir: str | Path, short_name: str) -> list[dict]:
    """Write one PDF per chunk into out_dir, named <short_name>__NN__<title>.pdf.
    Uses pypdf for lossless page copying. Returns metadata per written chunk."""
    from pypdf import PdfReader, PdfWriter

    p = plan(pdf_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader = PdfReader(str(pdf_path))
        written = []
        for i, (s, e, title) in enumerate(p["chunks"], 1):
            w = PdfWriter()
            for pg in range(s, e):
                w.add_page(reader.pages[pg])
            name = f"{short_name}__{i:02d}__{_slug(title)}.pdf"
            with open(out_dir / name, "wb") as fh:
                w.write(fh)
            written.append({"name": name, "start": s, "end": e, "pages": e - s, "title": title})
    return {"method": p["method"], "pages": p["pages"], "written": written}


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        pl = plan(arg)
        print(f"{arg}\n  {pl['pages']}pp via {pl['method']} -> {len(pl['chunks'])} chunks")
        for s, e, t in pl["chunks"]:
            print(f"    p{s+1:>4}-{e:<4} ({e-s:>2}pp)  {t[:60]}")
