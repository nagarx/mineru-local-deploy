# local_deploy — max-precision PDF → Markdown (Apple Silicon)

A thin, precision-obsessed batch layer over MinerU 3.4.4 for turning research-paper PDFs
into clean Markdown (body text + `$$LaTeX$$` equations + `<table>` HTML) for Claude Code
agents. **Precision over speed**; the pipeline is built to never *silently* drop a word,
number, equation, or table.

Full design + rationale: `../docs/superpowers/specs/2026-07-01-mineru-local-deployment-design.md`.

## What it does

For every PDF it runs **two** MinerU backends and reconciles them:

1. **hybrid-engine `--effort high`** (primary) — VLM (`MinerU2.5-Pro-2605-1.2B`, BF16 via MLX)
   for structure/equations/tables + the PDF's native text layer for character-exact body text.
2. **pipeline** (cross-check) — the deterministic, never-hallucinating backend.

Then, offline, it builds the Markdown from the hybrid `content_list.json` and runs guards:

| Guard | What it catches |
|---|---|
| Keep-list (R1) | keeps references (`ref_text`), footnotes (`page_footnote`), margin notes (`aside_text`); drops only figures + figure captions + page furniture |
| Table guard (R2) | a table that failed recognition (image-only / empty / raw OTSL tokens) is **flagged, never dropped** |
| Equation guard (R3) | an equation with no LaTeX body is flagged, never dropped |
| Coverage net (R7) | every number/word in the PDF text layer must appear in the output (numeral-recall is the primary gate) |
| Cross-check (R4/R6) | numbers/words the deterministic pipeline found but hybrid lacks are flagged |
| OCR-misclass (R5) | a born-digital PDF parsed as "scanned" is flagged (consider `-m txt`) |
| Empty-block salvage (R8) | a KEPT text block the VLM returned EMPTY is recovered from the PDF text layer via its bbox and marked `SALVAGED` — it used to drop silently, because the flag machinery only covered tables/equations/code |
| Text-layer repair (R9) | `??` where the VLM could not emit a maths glyph is restored from the text layer (`repair.py`) |
| Sign FLAG (R10) | **Changed 2026-08-12: this no longer repairs, it flags.** A minus dropped from a negative number used to be restored automatically. Value-based matching constrains the *evidence* but never the *target*, so it signed digits that were never negative — measured over 636 documents, ~10–20% of its edits were correct and it shipped `\frac {−1}{2}`, `N _ { −1 }`, the year `2011` and page ranges. Candidates are now recorded in `stats.sign_edits` with their context and counted in `signs_flagged`; nothing is applied. `MINERU_APPLY_SIGN_REPAIR=1` restores mutation. |
| Table shape (R11) | the two backends disagreeing on a table's **column count** is flagged — a recognizer that drops a whole column still emits valid HTML and passes every other guard |

Anything suspicious lands in a per-paper `*.qa.json` and the batch `report.md` for human review.

## Prerequisites (one-time, already done on this machine)

- `uv` venv on Python 3.12 at `../.venv`, MinerU installed editable with `[all]` (incl. `mlx-vlm`).
  Restore/extend the env ONLY with `uv pip install -e ".[all]"` — never a blind `uv sync` (see below).
- **Critical dependency pair — enforced in three layers, and DELIBERATELY TIGHTER THAN UPSTREAM.**
  Upstream 3.4.4 allows `pdftext<0.8` / `pypdfium2<6`; we do not. **Do not relax our bounds to
  match upstream's.** The two crashes that originally forced the pin *are* fixed in 3.4.4
  (`_get_pdfium_page_object_bounds`, `_ensure_legacy_chars`) — so anyone re-reading the old
  rationale will conclude the pin is obsolete. It is not; the reason changed:
  **`pdftext >=0.7` silently destroys every non-BMP character.** Its `get_chars` does
  `if 0xD800 <= code <= 0xDFFF: code = 0xFFFD`, treating UTF-16 surrogates as corruption. They
  are not — `FPDFText_GetUnicode` returns UTF-16 *code units*, so every codepoint above U+FFFF
  legitimately arrives as a **pair**. Measured on one CROSSFORMER page: pdftext 0.6.3 → 190
  recoverable maths glyphs; 0.7.1 → **380 `U+FFFD` and 0 recoverable**. Unlike the old failures
  this one does not crash — it just yields quietly wrong papers, and the loss happens *inside*
  pdftext where no downstream repair can reach it. (The earlier incident is still instructive:
  2026-07-08, a `uv` re-lock reverted a manual pin and cost a books run.) Enforcement:
  1. **`pyproject.toml` bounds** (`pypdfium2>=4.30.0,<5`, `pdftext>=0.6.3,<0.7`) — every resolver
     (uv / pip / fresh clone) lands on the compatible pair. pdftext pins pypdfium2 *exactly*
     (`0.6.3 → ==4.30.0`), so the two always move together; pypdfium2 is not independently choosable.
  2. **Runtime preflight** — `convert.preflight_deps()` aborts every model-running
     `run.py`/`convert.py` invocation loudly if the pair is wrong (2-second failure instead of a
     silently degraded run; offline paths like `--status`/`--verify`/`--rebuild` skip it by design).
     Its main check is **behavioural, not a version string**: it reads pdftext's `get_chars` and
     refuses any build that pre-empts surrogates, so the pin can be relaxed on evidence the day
     upstream fixes this properly. It also asserts the fork patch itself is still present.
  3. Manual recovery one-liner if an env ever drifts anyway:
     `uv pip install "pdftext==0.6.3" "pypdfium2>=4.30,<5"`
  There is deliberately **no `uv.lock`** in this repo: the one that briefly existed was a fresh
  resolution that diverged from the validated env on 12 packages (incl. torch), and syncing from
  it is exactly what caused the incident.
- Models downloaded (`mineru-models-download -s huggingface -m all`); paths in `~/mineru.json`.
- (Max precision) source DPI raised 200 → 300 in `mineru/utils/pdf_image_tools.py` and the
  render cap 3500 → 4500 in `mineru/utils/pdf_reader.py`.
- **UTF-16 surrogate fix in `mineru/utils/pdf_text_tool.py`** (`_merge_surrogate_pairs`).
  pdftext's `get_chars` does `chr(FPDFText_GetUnicode(textpage, i))`, but that PDFium call
  returns a UTF-16 **code unit**: every non-BMP character — which is all of Mathematical
  Alphanumeric Symbols (U+1D400–U+1D7FF), i.e. the italic 𝑥/𝑡/𝑋/𝜃 that maths papers use —
  arrived as two lone surrogates. Lone surrogates cannot be encoded, so downstream they were
  dropped outright or became `?` apiece, which is where the literal `??` in extracted body
  text came from. Until this fix the "character-exact native text" guarantee silently failed
  on maths-heavy papers (14 of 51 in the 2026-07-28 batch; 9,268 characters).
  An **unpairable** half (a genuinely broken ToUnicode CMap, e.g. ChronosX p3 where `U+D835`
  is followed by `!`) is unrecoverable, so it becomes `U+FFFD` — visible, and safe to encode,
  rather than crashing a writer or vanishing silently. That is pdftext 0.7's remedy applied
  only where it is *correct*; 0.7's bug is applying it to valid pairs as well.
  **This is a fork-local patch with no upstream equivalent, and upstream is unlikely to adopt
  it** — every surrogate issue in MinerU's tracker (#1203, #1546, #2525, #4685) was closed by
  *stripping* surrogates rather than decoding them. So:
  - **re-apply it after every `git merge upstream/master`** — 3.4.4 rewrote this file (+195
    lines) and the patch does not survive automatically;
  - `tests/local_deploy/test_surrogate_pairs.py` fails loudly if it is lost or if pdftext is
    upgraded past it — run it after any merge or dependency change;
  - do **not** "delete this patch because pdftext fixed it" without checking the test: 0.7
    *claims* to handle surrogates and in fact destroys them.

## Tracking upstream MinerU

We run a **fork**, currently merged up to upstream **3.4.4** (`git remote` `upstream` →
`opendatalab/MinerU`). The fork surface on upstream code is deliberately tiny — three files,
four patches — because `local_deploy/` does all its real work by shelling out to the `mineru`
CLI rather than calling its API. Keep it that way: it is what makes upstream merges cheap.

The one exception is deliberate: **preflight imports `mineru` to introspect it** — to assert
the fork patches are present, the render DPI is 300, the version is 3.4.x, and the `pipeline`
backend still exists. That is a read-only invariant check, not a coupling to MinerU's
internals, and it exists precisely so a lost patch fails loudly instead of silently degrading
every extraction.

| Upstream file | Our change | Merge risk |
|---|---|---|
| `mineru/utils/pdf_text_tool.py` | `_merge_surrogate_pairs` | **High** — upstream edits this file; re-apply every time |
| `mineru/utils/pdf_text_tool.py` | `_is_same_glyph_expansion` (ligature fix) | **High** — same file; re-apply every time |
| `mineru/utils/pdf_image_tools.py`, `mineru/utils/pdf_reader.py` | DPI 200→300, cap 3500→4500 | Low — upstream rarely touches these |
| `pyproject.toml` | tighter `pdftext`/`pypdfium2` bounds | **Always conflicts** — resolve in *our* favour |

**Ligature fix (2026-08-11).** Upstream's `_deduplicate_near_identical_chars` deletes a
visible character whose same character was already seen at a near-identical bbox. A
ligature is **one glyph** whose ToUnicode expands to several characters, all carrying that
glyph's bbox — so the second `f` of "different" was deleted, emitting `diferent`. Measured
before the fix: **267 of 636 documents (42%)** — and, robustly, **exclusively on the
native-text path: 267/551 native vs 0/85 VLM-OCR**, a total correlation reproduced
independently with different word lists.

> **Occurrence counts, corrected 2026-08-12.** Do not quote a single total. At least seven
> mutually inconsistent figures are on record (7,340 / 8,618 / 10,363 / 10,728 / 13,030 /
> 13,208 / 15,560) because each measurement used a different word list and none said so.
> The file *count* is stable at 265–277; the **split by parse path is the robust claim**.
> The measurement scripts are preserved at `tools/verification/v_lig{,2,3}.py`.
>
> `tt` collapse is real but **rare: 7 files / 9 occurrences**, not the "164 files" stated
> earlier — that regex included the ordinary English word "matter", which alone matched 157
> files. The patch is justified by the f-family numbers; `tt` shows the mechanism is not
> f-specific, not that it is common.

`_is_same_glyph_expansion` refuses the
deletion only for a constituent that is immediately adjacent in the kept stream with a
**bit-for-bit equal** bbox, which is the measured ligature signature; merely *near*-identical
bboxes and non-adjacent repeats still dedupe, so upstream's two real duplicate classes are
untouched. Report: `upstream_reports/mineru-ligature-dedup.md`.

**Both `pdf_text_tool.py` patches are asserted by `convert.preflight_deps()`** and pinned by
`tests/local_deploy/test_surrogate_pairs.py` and `test_ligature_dedup.py` — run both after
any merge. The DPI/render-cap patch is now asserted too (`preflight_fork_invariants()`); it
previously had **no** guard at all, so losing it silently halved VLM table-crop resolution.

**To merge a new upstream release:**
```bash
git fetch upstream --tags
git merge upstream/master              # expect exactly one conflict: pyproject.toml
# keep OUR pdftext/pypdfium2 bounds, take upstream's other dependency changes
.venv/bin/python tests/local_deploy/test_surrogate_pairs.py   # MUST pass
.venv/bin/python tests/local_deploy/test_ligature_dedup.py    # MUST pass
.venv/bin/python -c "import sys;sys.path.insert(0,'local_deploy');from convert import preflight_deps;print(preflight_deps() or 'OK')"
```

> **As of 2026-08-11 there is nothing to merge.** Verified live against the GitHub API and
> PyPI: 3.4.4 (2026-07-10) is the latest stable, `upstream/master` is byte-identical to our
> merge base (`ahead_by: 0, behind_by: 0`), and the 3.x line is frozen — all activity is on
> `next`, which is 4.0. Treat this runbook as dormant rather than routine.
Then re-extract **one** maths-heavy paper and diff it against the previous output before
trusting a batch run.

**Do not adopt MinerU 4.0.x.** It is a client/server rewrite that **deletes the `pipeline`
backend** (and `vlm`), which is what our two-phase hybrid↔pipeline cross-check compares
against — adopting it would silently remove a precision guarantee, not just change an API.
It also ships telemetry. Re-evaluate only if `pipeline` (or a real equivalent) returns.

> **Model-weight gotcha:** the HuggingFace download can report success while silently
> skipping a large weight file. `convert.py` runs `preflight_models()` and aborts before any
> phase if a weight is missing. To re-fetch a specific missing model:
> ```python
> from huggingface_hub import snapshot_download
> snapshot_download("opendatalab/PDF-Extract-Kit-1.0",
>                   allow_patterns=["models/MFR/unimernet_hf_small_2503/*"])
> ```

## Bulk library — two tracks (research papers + books), cyclic & resumable

Papers and books run as **two independent libraries** — separate inboxes, outputs, and ledgers,
selected with `--root`, so nothing is ever mixed:

```
library/
  research_papers/   ← TRACK 1: drop paper PDFs in inbox/ ;  paper .md in output/
  books/             ← TRACK 2: drop PRE-SPLIT chapter PDFs in inbox/ ;  chapter .md in output/
```

**Research papers** — drop PDFs straight into `library/research_papers/inbox/` (any number, anytime;
subfolders fine), then run with that track as `--root`:

```bash
R=local_deploy/library/research_papers
python local_deploy/run.py --root $R --loop         # process everything, cycle by cycle, until drained
python local_deploy/run.py --root $R                # one cycle (next --batch), then exit
python local_deploy/run.py --root $R --status       # counts + refresh report.md
python local_deploy/run.py --root $R --verify       # assert every 'done' paper has its .md
python local_deploy/run.py --root $R --retry-failed # requeue anything that failed, then cycle
python local_deploy/run.py --root $R --rebuild      # re-emit every .md from the .sidecar cache (offline)
```

**Books** — a whole book (300–500 pp) blows MinerU's ~1 h/document timeout, so **split each book into
≤55 pp chapter PDFs first** (`booksplit.py`), drop the chapters into `library/books/inbox/`, then run
the exact same commands with `--root local_deploy/library/books`. Each chapter becomes its own `.md`.
The pipeline can't "detect" a book — it treats every PDF identically; splitting is a *human* decision.

- **`--root` is required** — it names the track; there is no default, so you never run the wrong one.
- **Markdown out:** `<track>/output/<slug>.md` — the single, self-contained deliverable your agents read (clean text + `$$LaTeX$$` + `<table>` HTML, with `<!-- page N -->` markers and a leading `MINERU-QA` trust/provenance header). The structured JSON (`content_list.json`, `qa.json`) is an **operator-only cache** under `<track>/output/.sidecar/`, off the agent path — see "Why Markdown-only" below.
- **Never-miss:** every PDF is tracked in a per-track SQLite ledger by content-hash, so `done + pending + failed == total`, always. Duplicates (same content, any name) convert once; a moved/renamed PDF is never reprocessed.
- **Fully resumable:** kill it anytime and re-run — it resumes with nothing reprocessed and nothing missed. Safe to keep copying PDFs into a track's `inbox/` while it runs.
- **Robust:** a corrupt/poison PDF fails *in isolation* (its batch-mates still convert) and is listed in the track's `report.md` (never silently dropped); a file that's mid-copy/locked is skipped and retried next scan.
- **Isolated:** each track has its own ledger, lock, and `report.md` — run one track at a time on 16 GB (both backends are memory-heavy). Tuning: `--batch N` (PDFs/cycle, default 8), `--effort`, `--method`, `--window`.

> The completed first corpus (137 files) is archived intact at `library/_archive_first_corpus/` — still a runnable root: `run.py --root local_deploy/library/_archive_first_corpus --status`.

## One-off run (a single folder or file)

```bash
.venv/bin/python local_deploy/convert.py --input <dir-or-pdf> --output out
```

- `--input`  a PDF file or a directory (searched recursively)
- `--output` where `<stem>.md` + `report.md` go (agent-facing); the JSON cache lands in `<output>/.sidecar/`
- `--workdir` scratch dir for raw backend output (default `OUTPUT/_work`)
- `--phases`  `all` (default) or any of `hybrid,pipeline,build` (for resuming/re-analysis)
- `--effort`  `high` (default) | `medium`
- `--window`  `MINERU_PROCESSING_WINDOW_SIZE` (default 32; lower to 16 if you hit MPS OOM)

The two backend phases run as **separate subprocesses** so the VLM and pipeline model
stacks are never co-resident (safe on 16 GB). `build` loads no models. Re-running is cheap:
`--phases build` re-analyzes existing backend outputs without re-inferring.

## Reading the output

- `<stem>.md` — **the deliverable, and the only file agents read.** It opens with a
  `<!-- MINERU-QA … -->` header (source, page count, verdict, coverage/cross-check recall, and
  how many blocks are low-confidence), carries `<!-- page N -->` markers at page boundaries, and
  marks any table/equation that failed recognition inline with `<!-- ⚠ MINERU-FLAG … -->` — so
  nothing vanishes silently and the trust signal travels with the content.
- `.sidecar/<stem>.content_list.json` — **operator cache**, not for agents: the raw hybrid
  content_list; the re-render source for `run.py --rebuild` (regenerate every `.md` offline when
  the builder/policy improves, with no VLM re-run).
- `.sidecar/<stem>.qa.json` — **operator record**: full stats + flags + coverage + cross-check
  (the `.md` header is a distillation of this).
- `report.md` — batch roll-up; **the "Needs human review" section is the checklist.**

### Why Markdown-only (not the JSON) for agents

The `.md` is a faithful superset of the paper's *content* — every word, number, equation (LaTeX),
table (HTML), reference, footnote, and table caption is in it. `content_list.json` adds no content
an agent lacks; its "extra" is furniture we deliberately strip (headers / footers / page numbers),
figure captions we drop by policy, pixel `bbox`es, and dangling image paths — plus ~45% more
tokens. Feeding it to agents re-introduces exactly the noise the `.md` removes and is a
retrieval/duplication footgun. The one genuinely useful datum it had — the page number — is
preserved as `<!-- page N -->` markers; the one useful signal `qa.json` carried — which extraction
to distrust — is folded into the `.md` header. So the JSON stays as an operator cache, and agents
get one clean, self-describing file.

## Notes

- **Method:** runs `-m auto`, which picks per document: character-exact **native text** for clean
  born-digital text layers, **VLM OCR** otherwise (bad/rotated/garbled layers). Both modes are
  adversarially ground-truth-audited faithful, and the coverage net independently validates the body
  text against the PDF's own text layer either way, so any transcription error surfaces as a
  word-recall drop. (`-m txt`/`-m ocr` force a mode. The pdftext/pypdfium2 pin above is what keeps
  the native-text path character-exact — on pdftext 0.7.x every non-BMP maths glyph is destroyed
  before MinerU sees it; that is enforced behaviourally, not assumed.)
- **`needs_review` vs `notes`:** `review_reasons` are hard triggers (failed table/equation, low
  coverage/cross-check recall, a suspect page). `notes` are informational and do NOT trigger review
  (OCR mode; hybrid-vs-pipeline table-count differences from the pipeline over-splitting a merged
  table — verified benign on the test set).
- Speed: MLX runs one page-image at a time; expect several minutes per paper. That's the precision tax.
- Determinism: VLM decodes greedily (temperature 0) → identical output every run.
- Tables are HTML (`<table>`), equations LaTeX (`$$…$$` / `$…$`) — embedded verbatim.
- Cosmetic only (not content loss): bold/italic emphasis inside tables isn't preserved; LaTeX tokens
  may be internally spaced (renders identically in any MathJax/LaTeX engine).
