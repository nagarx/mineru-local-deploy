# Upstream report — MinerU

**Repo:** https://github.com/opendatalab/MinerU
**Affects:** 3.4.4 (introduced by `6826a9de`, "feat: add offset duplicate character detection for
improved text rendering", 2026-07-10, and the surrounding near-identical dedup work); present on
`upstream/master`
**Prepared:** 2026-08-11 — DRAFT FOR REVIEW, not yet filed

---

## Title

`_deduplicate_near_identical_chars` silently destroys ligatures on the native-text path —
`different` is extracted as `diferent`

## Summary

`mineru/utils/pdf_text_tool.py::_deduplicate_near_identical_chars` drops any visible character
whose *same* character has already been seen at a *near-identical* bbox on the page
(`NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE = 1.0`).

A typographic ligature — `ﬀ`, `ﬁ`, `ﬂ`, `ﬃ`, `ﬄ` — is a **single glyph**. PDFium reports its
constituent characters through its ToUnicode mapping, and because they all originate from that one
glyph they carry **the same bbox**. The deduplicator therefore sees the second `f` of `ff` as a
duplicate of the first and deletes it.

Every `ff`/`ffi`/`ffl` ligature in a born-digital PDF loses a character:

| source word | extracted |
|---|---|
| `different` | `diferent` |
| `efficient` | `eficient` |
| `coefficient` | `coeficient` |
| `difficult` | `dificult` |
| `diffusion` | `difusion` |
| `effective` | `efective` |

There is no warning, no exception and no counter. The words remain plausible English, so nothing
downstream detects them.

## Impact (measured)

A 636-document corpus (research papers + book chapters, LaTeX-typeset, `hybrid-engine -m auto`):

- **267 of 636 documents affected (42%)**, **8,618 occurrences**
- Split by parse method, the correlation is total:

| path | documents | affected | rate | occurrences |
|---|---|---|---|---|
| native text (`ocr_enable=false`) | 551 | **267** | **48%** | 8,618 |
| VLM OCR (`ocr_enable=true`) | 85 | **0** | **0%** | 0 |

- Most frequent corrupted forms: `diferent` (3,023), `eficient` (695), `diference` (688),
  `coeficients` (452), `coeficient` (448), `difusion` (370), `diferences` (357), `eficiency` (356),
  `efective` (323), `dificult` (286).

The defect is confined to the path that is *documented as character-exact*. On a clean text layer
`-m auto` correctly selects `txt`, so the OCR path — which is unaffected — never runs.

It also evades recall-based quality gates. On a sampled paper, word-recall against the PDF text
layer is **0.9807**; across the corpus the median is 0.988. Any threshold loose enough not to
produce constant false positives (we use 0.90) is far too loose to catch this.

## Reproduction

Any born-digital PDF containing an `ff` ligature:

```python
import pypdfium2 as pdfium
from pdftext.pdf.chars import get_chars
import mineru.utils.pdf_text_tool as ptt

doc = pdfium.PdfDocument("paper.pdf")
page = doc[0]
tp = page.get_textpage()
chars = ptt._ensure_legacy_chars(ptt.deduplicate_chars(get_chars(tp, page.get_bbox(), 0, True)))

before = "".join(c["char"] for c in chars)
after  = "".join(c["char"] for c in ptt._deduplicate_near_identical_chars(chars))

print(before.count("different"), after.count("different"))   # e.g. 2 -> 0
print(before.count("diferent"),  after.count("diferent"))    # e.g. 0 -> 2
```

Observed on one real page: `'izontally differentiable ('` → `'rizontally diferentiable ('`,
2,906 chars in, 2,904 out — exactly the two duplicate `f`s removed.

## Root cause

`_deduplicate_near_identical_chars` keys on `(visible char signature, bbox bucket)` and treats any
repeat as a rendering artifact:

```python
if any(
    _is_near_identical_bbox(bbox_coords, seen_bbox)
    for neighbor_bucket_key in _iter_neighbor_bbox_bucket_keys(bbox_bucket_key)
    for seen_bbox in visible_char_bbox_buckets.get(neighbor_bucket_key, [])
):
    continue          # <-- the second 'f' of a ligature dies here
```

The premise "same character + same position ⇒ duplicate" does not hold for multi-character glyph
mappings. It is the same structural issue MinerU already hit with UTF-16 surrogate pairs, where the
two halves likewise share one glyph box — see the note in this repo's
`mineru-surrogate-pairs.md`, whose fix comment already observes that *"halves share one bbox, so
deduping first can strip one half of a pair."* Ligatures are the same phenomenon with a
multi-character ToUnicode mapping instead of a multi-code-unit codepoint.

The `_is_adjacent_offset_duplicate_char` sibling check is **not** implicated: it requires a
diagonal translation (`abs(y_start_offset) > NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE`), which a
horizontally-set ligature does not have.

## Suggested fix

Do not deduplicate characters that originate from the same glyph. Two workable discriminators:

1. **Exact-bbox adjacency.** Ligature constituents are *consecutive in the char stream* and share an
   effectively identical bbox. Text-layer boundary duplicates — the artifact this function was added
   to remove — are repeats of a character already emitted **earlier and elsewhere** in the stream.
   Restricting the near-identical test so it never removes a char whose match is the immediately
   preceding stream entry with an identical box preserves ligatures while still clearing boundary
   duplicates.
2. **Ligature-aware guard.** Skip the drop when the run of same-bbox characters spells a known
   ligature expansion (`ff`, `fi`, `fl`, `ffi`, `ffl`, and the `ﬅ`/`ﬆ` forms).

Either keeps the intended behaviour — removing genuinely duplicated *rendered* characters — while
not conflating "one glyph, several characters" with "one character, drawn twice".

Neither discriminator has been implemented or validated yet — the analysis above is diagnosis only.
We intend to carry a fix as a fork-local patch alongside `_merge_surrogate_pairs`, with a regression
test and a corpus-wide validator that fails if any collapsed-ligature form reappears, and will
update this report with measured before/after numbers once that exists. Happy to open a PR then.
