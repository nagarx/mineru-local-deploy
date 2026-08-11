# Upstream report — MinerU

**Repo:** https://github.com/opendatalab/MinerU
**Affects:** 3.4.4 and earlier (checked through `upstream/master` @ 3.4.4); 4.0.0a4 inherits the same code path
**Prepared:** 2026-07-29 — DRAFT FOR REVIEW, not yet filed

---

## Title

Non-BMP mathematical variables are extracted as `??` — surrogate pairs are never recombined

## Summary

`mineru/utils/pdf_text_tool.py::get_page_chars` consumes pdftext's `get_chars` output directly.
On pdftext 0.6.x each character arrives as a UTF-16 **code unit**, so every codepoint above
U+FFFF is delivered as two lone surrogates. MinerU never recombines them, and because a lone
surrogate cannot be UTF-8 encoded, each one is dropped or rendered as `?` further down — so a
single mathematical variable becomes the literal string `??` in the extracted Markdown.

This silently breaks the character-exact native-text guarantee on exactly the documents where
it matters most: the affected range is the **Mathematical Alphanumeric Symbols** block
(U+1D400–U+1D7FF), which is what LaTeX emits for italic maths variables.

## Impact (measured)

On a 51-paper corpus of LaTeX-typeset ML papers extracted with `hybrid-engine -m auto`:

- **1,187** `??` occurrences in the produced Markdown
- **14 of 51** papers affected, 9,268 characters
- corruption concentrated on variable names — `??` where the source reads `𝑋`, `𝜃`, `𝑡`

Because the text layer is otherwise clean, `-m auto` correctly selects `txt`, so the OCR path
never runs and nothing flags the loss. There is no warning, no exception, and no recall drop —
the characters simply cease to exist.

## Reproduction

`repro_surrogates.py` in this directory demonstrates the pdftext-level behaviour on any
LaTeX-typeset PDF. At MinerU level:

```python
import pypdfium2 as pdfium
from mineru.utils.pdf_text_tool import get_page_chars

doc = pdfium.PdfDocument("paper.pdf")
text = "".join(c["char"] for c in get_page_chars(doc[4])["chars"])
print(sum(1 for ch in text if 0xD800 <= ord(ch) <= 0xDFFF))   # > 0 on any maths paper
text.encode("utf-8")                                           # UnicodeEncodeError
```

## Root cause

`FPDFText_GetUnicode` returns a UTF-16 code unit. pdftext's `get_chars` does `chr(...)` on it
(0.6.x), so pairs arrive split. This is fundamentally a pdftext issue — a separate report is
being filed there — but MinerU is the layer that can still repair it losslessly, and MinerU is
where the corruption becomes user-visible.

Note that pdftext >=0.7 does **not** fix this: it replaces every surrogate with `U+FFFD`,
including valid pairs, which converts recoverable corruption into unrecoverable loss. MinerU
3.4.4's `pdftext>=0.6.3,<0.8.0` bound therefore spans both a repairable and an unrepairable
backend. Related MinerU issues (#1203, #1546, #2525, #4685) were each resolved by *stripping*
surrogates, which removes the symptom and the data together.

## Suggested fix

Recombine pairs **before** `deduplicate_chars`, so dedup compares real characters (the halves
share one glyph box, so deduping first can strip one half and leave the other unpairable):

```python
chars = deduplicate_chars(
    _merge_surrogate_pairs(
        get_chars(textpage, page_bbox, page_rotation, quote_loosebox)
    )
)
chars = _ensure_legacy_chars(chars)
```

where `_merge_surrogate_pairs` decodes `high+low` into one char keeping the first half's
bbox/font verbatim (layout is unchanged), and maps a genuinely **unpairable** half — which does
occur, from broken ToUnicode CMaps — to `U+FFFD` so nothing un-encodable escapes.

We have run this in production over 302 documents: `??` went from 1,187 to **0**, with no
layout change and no regression in table/formula recognition. Full implementation (~45 lines,
Apache-2.0 compatible) is at `mineru/utils/pdf_text_tool.py::_merge_surrogate_pairs` in our
fork, with tests at `tests/local_deploy/test_surrogate_pairs.py`. Happy to open a PR.
