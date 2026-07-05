# MinerU Local Deployment — Design Spec

**Date:** 2026-07-01
**Status:** Approved (pending adversarial validation of this design)
**Target machine:** Apple M1 Pro, 16 GB unified memory, macOS 26.2 (arm64)
**Goal:** Maximum-precision PDF → Markdown extraction of research papers (text, equations in LaTeX, tables) for Claude Code agent consumption. **Precision >> speed.** Must not miss or alter a single word, number, equation, or table.

---

## 1. Context

MinerU 3.4.0 (cloned from `opendatalab/MinerU`, `master`) converts PDFs to Markdown. It is used to build a corpus that feeds Claude Code agents working on the user's HFT/quant codebase. The dominant input is born-digital research papers (arXiv/journal), primarily English, with equations and tables.

## 2. Machine & environment constraints

- **16 GB unified memory** (shared CPU/GPU) — the binding resource. Mitigated by concurrency=1 and a reduced processing window.
- System `python3` is **3.14.2**, which MinerU rejects (`requires-python >=3.10,<3.14`). We use an isolated **uv venv on Python 3.12.12** (already installed; conservative wheel compatibility).
- No CUDA. Apple Silicon acceleration via **MLX** (VLM) and **MPS** (pipeline torch models). ONNX table models run **CPU-only** on Mac (no MPS/CoreML EP) — a speed cost, not a precision one.
- `uv` is the mandated package manager.

## 3. Key technical findings (drive the decisions)

Established by three independent code-exploration agents over the MinerU 3.4.0 source (see conversation history for file:line evidence):

1. **Backends:** `pipeline` (classic small models, OmniDocBench 86.47), `vlm-engine` (pure 1.2B VLM, 95.30), **`hybrid-engine` (default; VLM + pipeline native-text + sidecars, 95.39 at `--effort high`)**. `*-http-client` variants target remote OpenAI servers (not used here).
2. **VLM model:** `opendatalab/MinerU2.5-Pro-2605-1.2B` — a **1.2B-param Qwen2-VL** derivative, **BF16** weights.
3. **MLX = full precision.** On macOS, if `mlx-vlm` is installed, MinerU auto-selects the MLX engine and loads the **same BF16 weights, unquantized** (no int4/int8). So the native/fast path is also the max-precision path. (int4 GGUF variants exist only for llama.cpp/Ollama; MinerU never uses them.)
4. **Deterministic decoding:** greedy (temperature 0, top-k 1). Same PDF → identical Markdown. Reproducible/auditable.
5. **`--effort high` is mandatory** on hybrid: `medium` silently disables image/chart analysis and uses pipeline-driven layout; `high` uses VLM-native layout + honors image analysis. (We disable image analysis anyway — see output policy — but keep `high` for the superior VLM layout/reading-order.)
6. **`lang` is ignored by hybrid/VLM** (pipeline-OCR only). Default `ch` already covers English/Latin; there is no separate `en` model. Nothing to tune for English on the primary path.
7. **Equations → LaTeX** (`$…$` inline, `$$…$$` block; `text_format: "latex"`). **Tables → HTML** (`<table>` with real rowspan/colspan). Both toggled on by default; hybrid force-enables formulas.
8. **DPI is hardcoded at 200** (`mineru/utils/pdf_image_tools.py:35`, capped at 3500 px long edge in `pdf_reader.py`) — not exposed as any flag/env var. Raising it requires a source edit (we have an editable install).
9. **Memory:** VLM BF16 ≈ 2.4 GB; hybrid additionally loads pipeline models (layout/OCR/UniMerNet/table). Both stacks resident ≈ 6–7 GB — comfortable in 16 GB at concurrency 1. MLX runs **one page-image at a time, serially** (speed cost only).
10. **Orchestration:** the `mineru` CLI is a thin client over a local `mineru-api` FastAPI service (auto-spawned unless `--api-url` given). Device/VRAM/model-source are **env-var-only** now (no CLI flags).

## 4. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Package manager | **uv** | Mandated; also manages the Python version. |
| Python | **3.12.12** (uv venv) | System 3.14 unsupported; 3.12 already installed, conservative wheels. |
| Install | **editable from source** `uv pip install -e ".[all]"` | Enables the DPI source edit; `[all]` pulls `mlx-vlm`+`mlx` on darwin + both model stacks. |
| Primary backend | **`hybrid-engine --effort high`** | Top accuracy (95.39) **and** native-text extraction → body text lifted character-exact from the PDF text layer (zero hallucination), VLM only for equations/tables/layout. |
| Cross-check backend | **`pipeline`** | Deterministic, never hallucinates; second opinion for the dual-backend diff (and covers scanned pages with no text layer). |
| Precision edit | **DPI 200 → 300**, long-edge cap 3500 → ~4200 | Sharpens formula crops + pipeline OCR; helps scanned pages. (VLM downsizes to ~1036 px, so this mainly benefits the pipeline sidecars — not oversold.) |
| Output policy | **Strictly text + LaTeX equations + HTML tables.** Drop all images/charts **and captions**. | User requirement. Markdown is **built by us from `content_list.json`**, keeping only `text`/`title`/`equation`/`table`/`list`/`code` blocks. |
| Artifacts per paper | `‹name›.md`, `‹name›.content_list.json`, `‹name›.qa.json` | Clean MD + structured blocks + QA report. |
| Workflow | **Warm local `mineru-api`** (`--enable-vlm-preload`) + **batch driver** over a folder | Model loads once, stays hot; idempotent/resumable batch. |
| Memory guards | `MINERU_API_MAX_CONCURRENT_REQUESTS=1`, `MINERU_PROCESSING_WINDOW_SIZE=32` | Safe peak memory on 16 GB. |
| Verification | **Dual-backend cross-check + text-coverage safety net** | User's hard "don't miss a number" requirement. |
| Model source | `huggingface` (auto) | Default; user has HF access. |

## 5. Architecture (components)

1. **Environment** — uv venv (Py 3.12) + editable `.[all]` install; sanity gate (`import mlx_vlm`, MPS available, `mineru --version`).
2. **One-time prep** — DPI/cap source edit; `mineru-models-download -s huggingface -m all` (both stacks); config written to `~/mineru.json`.
3. **Warm server** (`local_deploy/serve.sh`) — `mineru-api` with VLM preload, concurrency 1, window 32.
4. **Batch driver** (`local_deploy/convert.py`) — per PDF (idempotent/resumable):
   - **Pass A (primary):** hybrid-engine, effort high, formula+table on, image-analysis off → `content_list.json` + `middle.json` + server `md` (server md kept only as a reference to validate our builder).
   - **Pass B (cross-check):** pipeline backend → `content_list.json`.
   - **Text-coverage net:** extract PDF text layer (`pdftext`/`pypdf`); assert ~all tokens present in Pass A; flag low-coverage/empty pages.
   - **A↔B diff:** per-page normalized-text comparison; flag divergences.
   - **Clean-MD builder:** assemble Markdown from Pass A `content_list.json` (headings by `text_level`; equation `text` verbatim LaTeX; table `table_body` raw HTML; lists/code); drop image/chart/caption blocks. Diff our MD's text against the server md as a builder self-check.
5. **Outputs & batch report** — per-paper artifacts + a batch `report.md` listing papers needing human review.
6. **Error handling** — per-paper retry/timeout; corrupt/encrypted PDFs flagged, never crash the batch; server health check.

## 6. Validation strategy (adversarial, continuous)

Per user mandate — **independent "fresh-eyes" adversarial agents at every milestone**, not just at the end:

- **M0 (this doc):** red-team the design by re-reading the code — challenge that hybrid runs on MLX, that BF16 is loaded, that memory fits, that the DPI edit is safe, and that building our own MD from `content_list.json` cannot silently drop content the server MD contains.
- **M1 (env):** verify the install actually selects MLX (not transformers fallback) and loads the 1.2B model.
- **M2 (driver code):** adversarial code review of `convert.py` (silent-failure hunt, off-by-one page ranges, content-drop in the MD builder).
- **M3 (extraction):** after running the 3 test papers, an adversarial agent compares each source PDF against the produced `.md` to hunt for any missed/altered text, number, equation, or table.
- **M4 (tuning):** `medium` vs `high`, and DPI 200 vs 300, spot-compared on a real paper.

Test corpus: `research_paper_for_testing/` (3 born-digital quant/ML papers).

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| MLX fallback to slow transformers (mlx-vlm missing/broken) | M1 gate explicitly asserts MLX engine selected. |
| Our MD builder drops content present in server MD | Keep server MD; diff text; adversarial M0/M2 checks. |
| DPI edit breaks a 200-DPI assumption downstream (coords, caps) | Adversarial M0 review of all DPI/scale call sites; validate on test papers. |
| 16 GB OOM under both stacks | concurrency 1, window 32; monitor RSS during M1/M3. |
| pipeline `content_list` schema differs from hybrid → diff false positives | Normalize to text-only tokens before diffing. |
| Scanned pages (no text layer) escape the coverage net | Dual-backend A↔B diff covers these. |

## 8. Out of scope

Remote/OpenAI-server backends, Docker, multi-GPU/router, Gradio UI, the optional external-LLM title-hierarchy aid (`llm-aided-config`), non-Latin-script tuning. Can revisit later.

---

## 9. M0 red-team outcomes & revisions (2026-07-01)

Two independent adversarial agents re-read the code. **Install verified** (uv venv Py 3.12.12; torch 2.12.1 + MPS, mlx 0.31.1 / mlx-vlm 0.3.9, mineru-vl-utils 1.0.5, onnxruntime 1.27.0, transformers 4.57.6; all entry points work).

**Confirmed-safe (core approach stands):** `hybrid-engine`→`mlx-engine` routing on Darwin (no CUDA/vllm reachable); `image_analysis=false` + `--effort high` valid (disables only figure/chart description, never text/table/formula); DPI edit is **coordinate-safe** (per-image `scale` propagated consistently; single edit at `pdf_image_tools.py:35` suffices for PDF inputs; cap `pdf_reader.py:14` 3500→4200 is the sole other site); macOS "26.2" ≥ "13.5" parses correctly (PEP 440); `[all]` installs mlx on darwin; no int4/int8 quantization anywhere; **hybrid-high lifts body text character-exact from the PDF text layer for born-digital (identical to medium on characters — high differs only in layout quality + image-analysis).**

**Critical revisions (silent content-loss fixes — MUST implement):**

- **R1 — Expanded, per-backend keep-list.** A naive `{text,title,equation,table,list,code}` filter silently drops `ref_text` (**the entire bibliography**), `page_footnote`, `aside_text`, `phonetic`. Also VLM emits titles as `type:"text"` **with `text_level`**, not `type:"title"`. **Keep:** text (title rides here), equation, table (+ `table_caption`/`table_footnote`), list, code (+ `code_caption`), `ref_text`, `page_footnote`, `aside_text`, `phonetic`. **Drop only:** image, chart, `image_caption`, `chart_caption`, and (as noise) `header`/`footer`/`page_number`. Pipeline vs VLM emit these types **differently** (e.g. pipeline `ref_text`→`list`+sub_type) → filter **per-backend**.
- **R2 — Failed-table guard (CRITICAL).** A table that fails recognition can survive as `{type:"table", img_path, table_body:""}` (empty), or with raw OTSL tokens (`<fcel>…<nl>`) instead of HTML, or truncated (`finish_reason=length`) partial HTML. Since we drop images, an empty/img-only table = **whole table lost**. **Never drop:** flag any `type:"table"` whose `table_body` is missing / lacks `<table` / contains residual OTSL tokens / is malformed → route to pipeline cross-check + human review; compare row/cell counts vs pipeline.
- **R3 — Failed-equation guard.** Flag any `type:"equation"` with missing `text` (pipeline: img-only) or empty `$$…$$` body (VLM).
- **R4 — Orphan-text drop.** Hybrid never calls `remaining_spans()`, so native text not covered by a detected layout block is dropped with no catch-all. The **coverage net + pipeline cross-check are mandatory**, not optional; prefer pipeline text where hybrid coverage < pipeline.
- **R5 — `_ocr_enable` check.** Born-digital PDFs can misclassify as scanned (CID/LaTeX-subset fonts, math-heavy pages) → text routed through the VLM raster (fidelity risk). Assert `middle.json._backend`/`_ocr_enable==false` for born-digital; if a trusted text-layer PDF classifies as OCR, flag and optionally force `parse_method="txt"`.
- **R6 — Memory: phase the two backends.** Do NOT co-reside VLM (~2.5 GB, never evicted once preloaded) + pipeline table stack on 16 GB. **Phase A:** warm `mineru-api` (VLM preload) → all PDFs (hybrid). **Phase B:** stop it, warm `mineru-api` (no preload) → all PDFs (pipeline). **Phase C:** offline analysis/build (no models). `window=32` (→16 if MPS alloc fails). `MINERU_API_MAX_CONCURRENT_REQUESTS` is a **no-op on macOS** (hard-pinned to 1) — rely on that.
- **R7 — Coverage-net normalization.** Compare pdfium-text vs extracted, both normalized: undo full→half width, markdown-escaping backslashes, de-hyphenation (soft hyphen U+00AD + line-break rejoin); NFKC + ligature expansion (fi/fl/ffi); collapse whitespace; **exclude equation/table regions from the text check** (they're LaTeX/HTML); if pdfium returns >5% PUA codepoints → flag "untrustworthy source," not "miss." **Numeral-recall** (every digit run present) is the primary "don't miss a number" gate.
- **R8 — Verify BF16.** MLX loads whatever dtype is on disk (no forced bf16). After download, confirm the VLM model dir is BF16 safetensors (not a `-4bit`/`-8bit` conversion).
- **R9 — Build source.** Assemble from `content_list.json` (text + reading order) + `middle.json` (span `html`/`score`/`img_path` for the R2/R3 guards) + the pipeline pass (R4). **Do NOT use the server `.md`** (bakes in image refs, no failure hooks).
- **R10 — Advanced lever (deferred).** VLM input resolution is capped by the model's baked-in `preprocessor_config.json` `max_pixels` (no env/config knob). Raising it there can improve dense-table/subscript fidelity — test carefully after the baseline works.
