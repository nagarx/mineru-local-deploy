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
- **Critical dependency pin (do this after install):** `pdftext 0.7.0` pulls `pypdfium2 5.9.0`,
  which is incompatible with MinerU 3.4.0 (it removed `PdfImage.get_pos` and made `PageChars`
  non-iterable) and **crashes on some PDFs** (and breaks `-m txt`). Pin the compatible pair:
  ```bash
  uv pip install "pdftext==0.6.3" "pypdfium2>=4.30,<5"
  ```
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

## Bulk library — 200+ PDFs, cyclic & resumable (recommended)

Drop PDFs into `local_deploy/library/inbox/` (any number, anytime; subfolders fine), then:

```bash
python local_deploy/run.py --loop         # process everything, cycle by cycle, until drained
# or one cycle at a time:
python local_deploy/run.py                # next 8 PDFs, then exit (repeat whenever)
python local_deploy/run.py --status       # counts + refresh report.md
python local_deploy/run.py --verify       # assert every 'done' paper has its .md
python local_deploy/run.py --retry-failed # requeue anything that failed, then cycle
python local_deploy/run.py --rebuild      # re-emit every .md from the .sidecar cache (offline, no models)
```

- **PDFs in:** `library/inbox/`.  **Markdown out:** `library/output/<slug>.md` — the single, self-contained deliverable your agents read (clean text + `$$LaTeX$$` + `<table>` HTML, with `<!-- page N -->` markers and a leading `MINERU-QA` trust/provenance header). The structured JSON (`content_list.json`, `qa.json`) is an **operator-only cache** under `library/output/.sidecar/`, off the agent path — see "Why Markdown-only" below.
- **Never-miss:** every PDF is tracked in an SQLite ledger by content-hash, so `done + pending + failed == total`, always. Duplicates (same content, any name) convert once; a moved/renamed PDF is never reprocessed.
- **Fully resumable:** kill it anytime and re-run — it resumes with nothing reprocessed and nothing missed. Safe to keep copying PDFs into `inbox/` while it runs.
- **Robust:** a corrupt/poison PDF fails *in isolation* (its batch-mates still convert) and is listed in `report.md` (never silently dropped); a file that's mid-copy/locked is skipped and retried next scan.
- **Status:** `library/report.md` (totals + review/failed lists). Only one `run.py` per library at a time (lock-enforced).
- Tuning: `--batch N` (PDFs/cycle, default 8), `--effort`, `--method`, `--window`.

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

- **Method:** runs `-m auto`, which classifies these (math-heavy) born-digital papers as OCR and
  reads body text with the VLM. Native-text mode (`-m txt`, character-exact) is currently broken on
  pypdfium2 5.9.0 (`'PageChars' object is not iterable`), so we rely on OCR — and the coverage net
  independently validates the OCR text against the PDF's own text layer, so any OCR transcription
  error surfaces as a numeral/word-recall drop. A page-by-page adversarial audit of the OCR output
  on the test set found **zero** content loss or transcription errors.
- **`needs_review` vs `notes`:** `review_reasons` are hard triggers (failed table/equation, low
  coverage/cross-check recall, a suspect page). `notes` are informational and do NOT trigger review
  (OCR mode; hybrid-vs-pipeline table-count differences from the pipeline over-splitting a merged
  table — verified benign on the test set).
- Speed: MLX runs one page-image at a time; expect several minutes per paper. That's the precision tax.
- Determinism: VLM decodes greedily (temperature 0) → identical output every run.
- Tables are HTML (`<table>`), equations LaTeX (`$$…$$` / `$…$`) — embedded verbatim.
- Cosmetic only (not content loss): bold/italic emphasis inside tables isn't preserved; LaTeX tokens
  may be internally spaced (renders identically in any MathJax/LaTeX engine).
