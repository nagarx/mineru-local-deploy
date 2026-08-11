# Upstream report — pdftext

**Repo:** https://github.com/datalab-to/pdftext
**Affects:** 0.6.x (silent corruption) and 0.7.x (irrecoverable data loss)
**Prepared:** 2026-07-29 — DRAFT FOR REVIEW, not yet filed

---

## Title

`get_chars` treats UTF-16 surrogate pairs as corruption, destroying every non-BMP character

## Summary

`FPDFText_GetUnicode` returns a UTF-16 **code unit**, not a codepoint. Every character above
U+FFFF is therefore delivered as a *valid surrogate pair* across two consecutive indices.
pdftext never recombines them:

- **0.6.x** calls `chr()` on each unit, emitting two lone surrogates. These cannot be UTF-8
  encoded, so they crash writers (`json.dump`, `Path.write_text`) or degrade to `?` each —
  producing a literal `??` in place of every affected character.
- **0.7.x** replaces each half with `U+FFFD`, commenting them as "lone surrogates (from broken
  ToUnicode CMaps)". This misdiagnoses valid pairs as corruption and makes the loss
  **irrecoverable** — the original codepoint is gone before any caller can see it.

The affected range is not exotic: it is the entire **Mathematical Alphanumeric Symbols** block
(U+1D400–U+1D7FF), which is what LaTeX emits for italic maths variables — the `𝑥`, `𝑡`, `𝑋`, `𝜃`
of essentially every mathematical paper.

## Impact

Measured on one page (p4) of an ICLR 2023 paper typeset in LaTeX:

| | pdftext 0.6.3 | pdftext 0.7.1 |
|---|---|---|
| lone high surrogates | 190 | 0 |
| lone low surrogates | 190 | 0 |
| `U+FFFD` emitted | 0 | **380** |
| correctly decoded non-BMP chars | 0 | 0 |
| **recoverable by the caller** | **190** | **0** |

Across a 51-paper corpus this produced 1,187 `??` occurrences in extracted text before we
patched it downstream. Because the characters are *variables*, the corruption lands precisely
on the semantically load-bearing tokens — `??` where the paper said `𝑋`.

## Reproduction

`repro_surrogates.py` (in this directory) runs against any LaTeX-typeset PDF containing italic
maths and prints the decoded/destroyed counts for the installed pdftext version.

```
python repro_surrogates.py <paper.pdf> <page-index>
```

The underlying call is direct and easy to confirm without pdftext at all:

```python
import pypdfium2 as pdfium, pypdfium2.raw as pdfium_c
tp = pdfium.PdfDocument("paper.pdf")[4].get_textpage()
codes = [pdfium_c.FPDFText_GetUnicode(tp.raw, i) for i in range(tp.count_chars())]
# ... 0xD835, 0xDC4B, ...   ->  0x1D44B  '𝑋'  MATHEMATICAL ITALIC CAPITAL X
```

## Why the 0.7 behaviour is not a fix

Lone surrogates *do* also occur for the stated reason — genuinely broken ToUnicode CMaps. We
observe those too (e.g. a high surrogate followed by `!`). The two cases are trivially
separable, and only the second is unrecoverable:

- **high followed by low** → a valid pair → decode it
- **anything else** → genuinely broken → `U+FFFD` is the right answer

0.7 applies the second rule to both cases. Note that `PageChars.codes` is already `np.uint32`,
which suggests full codepoints were the intent.

## Suggested fix

Decode pairs while scanning, and reserve `U+FFFD` for halves that cannot be paired. Sketch
against 0.7's `get_chars` loop (the second half contributes no row, so all per-char arrays stay
aligned):

```python
i = 0
while i < n:
    code = get_unicode(textpage_raw, i)
    if 0xD800 <= code <= 0xDBFF and i + 1 < n:
        low = get_unicode(textpage_raw, i + 1)
        if 0xDC00 <= low <= 0xDFFF:
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
            # emit ONE char; the pair shares a glyph box, so keep index i's box/font
            _emit(code, i)
            i += 2
            continue
    if 0xD800 <= code <= 0xDFFF:
        code = 0xFFFD          # genuinely unpairable — broken ToUnicode CMap
    _emit(code, i)
    i += 1
```

We run the equivalent of this downstream and it recovers all 190 glyphs on the page above, with
no change to layout (the two halves share one glyph box, so the first half's box is exact).

I'm happy to open a PR with tests if that would be useful.
