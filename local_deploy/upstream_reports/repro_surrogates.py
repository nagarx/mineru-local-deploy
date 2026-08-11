#!/usr/bin/env python
"""Self-contained reproduction for the pdftext / MinerU surrogate-pair bug.

Usage:
    python repro_surrogates.py <paper.pdf> [page-index]

Works against ANY LaTeX-typeset PDF that uses italic mathematical variables. Prints what the
installed pdftext version does with non-BMP characters, and what a caller could still recover.

Expected results:
    pdftext 0.6.x -> halves are emitted as lone surrogates; RECOVERABLE by merging pairs
    pdftext 0.7.x -> halves are replaced with U+FFFD inside get_chars; NOT recoverable
    fixed         -> characters already decoded; nothing to recover
"""
import sys

import pypdfium2 as pdfium
from pdftext.pdf.chars import get_chars

PDF = sys.argv[1]
PAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 0

try:
    import importlib.metadata as md
    print(f"pdftext={md.version('pdftext')}  pypdfium2={md.version('pypdfium2')}")
except Exception:
    pass

doc = pdfium.PdfDocument(PDF)
page = doc[PAGE]
raw = get_chars(page.get_textpage(), list(page.get_bbox()), page.get_rotation(), True)

# normalise 0.6 (list of dicts) and 0.7 (columnar PageChars) to a list of 1-char strings
chars = list(raw.text) if hasattr(raw, "text") else [c.get("char", "") for c in raw]

hi = sum(1 for c in chars if len(c) == 1 and 0xD800 <= ord(c) <= 0xDBFF)
lo = sum(1 for c in chars if len(c) == 1 and 0xDC00 <= ord(c) <= 0xDFFF)
fffd = sum(1 for c in chars if c == "�")
ok = sum(1 for c in chars if len(c) == 1 and ord(c) > 0xFFFF)

print(f"  page {PAGE}: {len(chars)} chars")
print(f"  lone high surrogates : {hi}")
print(f"  lone low  surrogates : {lo}")
print(f"  U+FFFD replacements  : {fffd}")
print(f"  already-decoded non-BMP chars: {ok}")

# what a caller can still salvage by recombining adjacent halves
out, i = [], 0
while i < len(chars):
    c = chars[i]
    if len(c) == 1 and 0xD800 <= ord(c) <= 0xDBFF and i + 1 < len(chars):
        nxt = chars[i + 1]
        if len(nxt) == 1 and 0xDC00 <= ord(nxt) <= 0xDFFF:
            out.append(chr(0x10000 + ((ord(c) - 0xD800) << 10) + (ord(nxt) - 0xDC00)))
            i += 2
            continue
    out.append(c)
    i += 1

rec = [c for c in out if len(c) == 1 and ord(c) > 0xFFFF]
print(f"  RECOVERABLE by merging pairs : {len(rec)}")
if rec:
    uniq = sorted(set(rec))
    print(f"    distinct glyphs : {''.join(uniq[:40])}")
    print(f"    codepoints      : {[hex(ord(c)) for c in uniq[:10]]}")
elif fffd:
    print("    -> destroyed inside get_chars; the original codepoints are unrecoverable")

doc.close()
