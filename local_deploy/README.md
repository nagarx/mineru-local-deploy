# local_deploy — max-precision PDF → Markdown (Apple Silicon)

A thin, precision-obsessed batch layer over MinerU 3.4.0 for turning research-paper PDFs
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

Anything suspicious lands in a per-paper `*.qa.json` and the batch `report.md` for human review.

## Prerequisites (one-time, already done on this machine)

- `uv` venv on Python 3.12 at `../.venv`, MinerU installed editable with `[all]` (incl. `mlx-vlm`).
  Restore/extend the env ONLY with `uv pip install -e ".[all]"` — never a blind `uv sync` (see below).
- **Critical dependency pair — enforced in three layers.** `pdftext 0.7.x` / `pypdfium2 5.x` break
  MinerU 3.4.0 twice over: 5.x removed `PdfImage.get_pos`, so `pdf_classify` crashes and **silently
  degrades every document to forced-OCR**; and 0.7.x returns a non-iterable `PageChars`, so any
  `txt`-classified document **crashes both backends deterministically**. This regressed once
  (2026-07-08: a `uv` re-lock reverted a manual pin and cost a books run), so it is now enforced:
  1. **`pyproject.toml` bounds** (`pypdfium2>=4.30.0,<5`, `pdftext>=0.6.3,<0.7`) — every resolver
     (uv / pip / fresh clone) lands on the compatible pair.
  2. **Runtime preflight** — `convert.preflight_deps()` aborts every model-running
     `run.py`/`convert.py` invocation loudly if the pair is wrong (2-second failure instead of a
     silently degraded run; offline paths like `--status`/`--verify`/`--rebuild` skip it by design).
  3. Manual recovery one-liner if an env ever drifts anyway:
     `uv pip install "pdftext==0.6.3" "pypdfium2>=4.30,<5"`
  There is deliberately **no `uv.lock`** in this repo: the one that briefly existed was a fresh
  resolution that diverged from the validated env on 12 packages (incl. torch), and syncing from
  it is exactly what caused the incident.
- Models downloaded (`mineru-models-download -s huggingface -m all`); paths in `~/mineru.json`.
- (Max precision) source DPI raised 200 → 300 in `mineru/utils/pdf_image_tools.py` and the
  render cap 3500 → 4500 in `mineru/utils/pdf_reader.py`.

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
  the native-text path working — on pdftext 0.7.x it crashes; that is enforced, not assumed.)
- **`needs_review` vs `notes`:** `review_reasons` are hard triggers (failed table/equation, low
  coverage/cross-check recall, a suspect page). `notes` are informational and do NOT trigger review
  (OCR mode; hybrid-vs-pipeline table-count differences from the pipeline over-splitting a merged
  table — verified benign on the test set).
- Speed: MLX runs one page-image at a time; expect several minutes per paper. That's the precision tax.
- Determinism: VLM decodes greedily (temperature 0) → identical output every run.
- Tables are HTML (`<table>`), equations LaTeX (`$$…$$` / `$…$`) — embedded verbatim.
- Cosmetic only (not content loss): bold/italic emphasis inside tables isn't preserved; LaTeX tokens
  may be internally spaced (renders identically in any MathJax/LaTeX engine).
