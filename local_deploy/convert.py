#!/usr/bin/env python3
"""
convert.py — Max-precision batch PDF -> Markdown for research papers (Apple Silicon).

Three memory-isolated phases (each backend runs as its own subprocess, so the VLM and
the pipeline model stacks are NEVER co-resident on 16 GB unified memory):

  hybrid    mineru -b hybrid-engine --effort high  ->  WORKDIR/hybrid/
  pipeline  mineru -b pipeline                      ->  WORKDIR/pipeline/   (deterministic cross-check)
  build     offline: clean Markdown + guards + coverage + cross-check (no models loaded)

Per-paper outputs:
  OUTPUT/<stem>.md                          the ONLY agent-facing file — clean, image-free
                                            Markdown (text + $$LaTeX$$ + <table>HTML), with
                                            `<!-- page N -->` markers and a leading MINERU-QA
                                            trust/provenance header.
  OUTPUT/.sidecar/<stem>.content_list.json  operator cache: the raw hybrid content_list; the
                                            re-render source for `run.py --rebuild` (no VLM).
  OUTPUT/.sidecar/<stem>.qa.json            operator record: stats + flags + coverage + cross-check.
OUTPUT/report.md            batch summary; explicitly lists papers that NEED HUMAN REVIEW.

The JSON sidecars add no content an agent lacks (the .md is a faithful superset of the paper's
text/equations/tables); they are operator assets kept OUT of the agent path on purpose.

Usage:
  convert.py --input DIR_OR_PDF --output OUTDIR [--workdir DIR] [--phases all|hybrid,pipeline,build]

Design rationale and the R1-R10 guards this enforces: see
docs/superpowers/specs/2026-07-01-mineru-local-deployment-design.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import postprocess  # noqa: E402
import coverage as cov  # noqa: E402
import crosscheck  # noqa: E402
import repair  # noqa: E402

PDF_SUFFIXES = {".pdf"}

# The review gate keys on RELIABLE signals only. Coverage-vs-text-layer NUMERAL recall is
# deliberately NOT a hard gate: it is confounded by figure axis-labels, glued table-cell
# digits, and rotated-table garbage text layers — all three test papers audited FAITHFUL
# despite low numeral recall (0.64-0.97). Reliable gates: recognition-failure guards,
# body-text WORD recall, word-based suspect pages, and the deterministic pipeline
# cross-check for numeric loss. Coverage numeral + its missing-number list are kept as
# INFORMATIONAL context for a human to eyeball / hand to an adversarial audit.
WORD_RECALL_MIN = 0.90         # body-text completeness vs the PDF text layer
XCHECK_NUMERAL_MIN = 0.90      # >=2-digit numbers the pipeline found but hybrid lacks
INFO_NUMERAL_RECALL = 0.97     # below this -> informational note (NOT a review trigger)


# --- helpers ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in PDF_SUFFIXES:
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in PDF_SUFFIXES)
    raise SystemExit(f"No PDF(s) at {input_path}")


def mineru_bin() -> str:
    cand = Path(sys.executable).parent / "mineru"
    return str(cand) if cand.exists() else "mineru"


_WEIGHT_EXTS = (".safetensors", ".pth", ".bin", ".onnx", ".pdparams")
_PIPELINE_SUBDIRS = [
    "models/Layout/PP-DocLayoutV2", "models/MFR/unimernet_hf_small_2503",
    "models/MFR/pp_formulanet_plus_m", "models/OCR/paddleocr_torch",
    "models/TabRec/SlanetPlus", "models/TabRec/UnetStructure",
    "models/TabCls/paddle_table_cls",
]


def _dir_has_weight(d: Path) -> bool:
    if not d.is_dir():
        return False
    for f in d.rglob("*"):
        if f.suffix.lower() in _WEIGHT_EXTS:
            try:
                if f.stat().st_size > 0:   # stat() follows symlinks; raises if dangling
                    return True
            except OSError:
                continue
    return False


def preflight_models() -> list[str]:
    """Verify the downloaded model weight files actually exist. The HuggingFace
    download can report success while silently skipping a large weight file — catch that
    here instead of crashing mid-run."""
    cfg = Path.home() / "mineru.json"
    if not cfg.exists():
        return ["~/mineru.json not found — run: mineru-models-download -s huggingface -m all"]
    md = json.loads(cfg.read_text()).get("models-dir", {})
    problems: list[str] = []
    pipe, vlm = md.get("pipeline"), md.get("vlm")
    if pipe:
        for sub in _PIPELINE_SUBDIRS:
            if not _dir_has_weight(Path(pipe) / sub):
                problems.append(f"pipeline model missing weights: {sub}")
    if vlm and not _dir_has_weight(Path(vlm)):
        problems.append("VLM model missing weights (model.safetensors)")
    return problems


def preflight_deps() -> list[str]:
    """Verify the pdftext/pypdfium2 pair this MinerU actually works with.

    HISTORY — the reason for this guard CHANGED at MinerU 3.4.4, do not "simplify" it back.
    It originally caught two crashes: pdf_classify dying on the removed PdfImage.get_pos
    (silently degrading EVERY doc to forced-OCR) and '-m txt' dying on a non-iterable
    PageChars. A uv re-lock regressed exactly that on 2026-07-08 and cost a books run.
    Upstream 3.4.4 FIXED BOTH (_get_pdfium_page_object_bounds, _ensure_legacy_chars), so
    those two symptoms can no longer be used to justify the pin.

    A third, worse reason now keeps it, and this one is silent rather than a crash:
    pdftext >=0.7 rewrites every UTF-16 surrogate to U+FFFD inside get_chars. Surrogates
    are not corruption — FPDFText_GetUnicode returns UTF-16 code units, so every codepoint
    above U+FFFF (the whole Mathematical Alphanumeric Symbols block: the italic x, t, X, θ
    of maths-heavy papers) legitimately arrives as a PAIR. Measured on one CROSSFORMER
    page: pdftext 0.6.3 -> 190 recoverable glyphs; 0.7.1 -> 380 U+FFFD and 0 recoverable.
    The loss happens inside pdftext, so nothing downstream can undo it.

    So the load-bearing check below is BEHAVIOURAL, not a version string: it asserts that
    this pdftext still hands us the surrogate halves. That keeps working if upstream ever
    fixes it properly and the pin can be relaxed on evidence rather than on a guess."""
    from importlib import metadata
    problems: list[str] = []

    def _ver(v: str) -> tuple[int, int, int]:
        """Numeric-prefix version triple; never raises ('4.30.0.post1' -> (4, 30, 0))."""
        out = []
        for tok in (v.split(".") + ["0", "0", "0"])[:3]:
            digits = ""
            for ch in tok:
                if not ch.isdigit():
                    break
                digits += ch
            out.append(int(digits) if digits else 0)
        return tuple(out)

    checks = (
        ("pypdfium2", lambda v: (4, 30, 0) <= _ver(v) < (5, 0, 0), ">=4.30,<5"),
        ("pdftext", lambda v: (0, 6, 3) <= _ver(v) < (0, 7, 0), ">=0.6.3,<0.7"),
    )
    for pkg, ok, want in checks:
        try:
            v = metadata.version(pkg)
            if not ok(v):
                problems.append(f"{pkg} {v} is incompatible with this MinerU (need {want})")
        except Exception as e:
            problems.append(f"{pkg} missing/unreadable: {e}")
    if not problems:
        # THE load-bearing check: behaviour, not version strings. A pdftext that pre-empts
        # surrogates destroys non-BMP maths silently — no crash, no log, just wrong papers.
        try:
            import inspect

            from pdftext.pdf import chars as _pdftext_chars
            _src = inspect.getsource(_pdftext_chars.get_chars)
            if "0xFFFD" in _src.upper().replace("0XFFFD", "0xFFFD"):
                problems.append(
                    "this pdftext replaces UTF-16 surrogates with U+FFFD inside get_chars — "
                    "every non-BMP mathematical variable would be destroyed before MinerU "
                    "can decode it, and no downstream repair can recover it")
        except Exception as e:
            problems.append(f"could not verify pdftext surrogate behaviour: {type(e).__name__}: {e}")
        try:
            import mineru.utils.pdf_text_tool as _ptt  # the fork patch must still be present
            if not hasattr(_ptt, "_merge_surrogate_pairs"):
                problems.append(
                    "mineru/utils/pdf_text_tool.py::_merge_surrogate_pairs is MISSING — the "
                    "fork patch was lost (most likely in a merge from upstream). Extracted "
                    "maths variables would silently become '??'. See tests/local_deploy/"
                    "test_surrogate_pairs.py")
            if not hasattr(_ptt, "_is_same_glyph_expansion"):
                problems.append(
                    "mineru/utils/pdf_text_tool.py::_is_same_glyph_expansion is MISSING — the "
                    "ligature fork patch was lost (most likely in a merge from upstream). "
                    "Every ff/fi/fl/tt ligature would silently lose a character "
                    "('different' -> 'diferent'), which affected 42% of the corpus before "
                    "the fix. See tests/local_deploy/test_ligature_dedup.py")
        except Exception as e:
            problems.append(f"pdf_text_tool import failed: {type(e).__name__}: {e}")
        problems += preflight_fork_invariants()
    if problems:
        problems.append('fix: uv pip install "pdftext==0.6.3" "pypdfium2>=4.30,<5"')
    return problems


def _lf(s: str) -> str:
    """Normalise PDFium's CRLF line endings to LF.

    `get_text_bounded()` joins lines with CRLF. `postprocess._recover_empty` emits that
    text verbatim into the deliverable and `_atomic_write` validates nothing, so 58 of
    636 shipped `.md` carry 552 raw CR bytes — an unflagged defect in an agent-facing
    file. It also makes the offline rebuild non-reproducible against the emitted corpus
    (244/302 match, 58 differ, all CR-only), which is what blocks a byte-identical
    regression gate. Normalising here fixes both at the source, before any consumer.
    """
    return s.replace("\r\n", "\n").replace("\r", "\n")


#: The render settings the fork patches in. They are SOURCE EDITS with no config path, so
#: nothing but an explicit assertion can notice their loss.
EXPECTED_DPI = 300
EXPECTED_RENDER_DEFAULTS = (300, 4500)


def fork_render_settings() -> dict[str, Any]:
    """The render invariants, for the preflight gate and the run manifest."""
    import mineru.utils.pdf_image_tools as _pit
    import mineru.utils.pdf_reader as _pr
    import mineru
    return {
        "mineru_version": getattr(__import__("mineru.version", fromlist=["__version__"]),
                                  "__version__", "?"),
        "mineru_path": str(Path(mineru.__file__).resolve().parent),
        "dpi": getattr(_pit, "DEFAULT_PDF_IMAGE_DPI", None),
        "page_to_image_defaults": tuple(_pr.page_to_image.__defaults__ or ()),
    }


def preflight_fork_invariants() -> list[str]:
    """Assert the fork's precision patches are actually loaded.

    ONLY `_merge_surrogate_pairs` was ever checked. The DPI/render-cap patch was
    protected by nothing, appears in ~0 of 636 QA records, and cannot be reasserted at
    runtime because MinerU renders in spawned workers that re-import from disk. Losing
    that one line silently renders everything at 200 DPI — and DPI is the ONE knob that
    matters, because every other consumer resizes to a fixed input: only the hybrid VLM's
    per-block crops scale with it (a table crop drops 2048 -> 903 visual tokens). Silent,
    undetectable after the fact, and it degrades exactly the equations and tables this
    pipeline exists to get right.
    """
    problems: list[str] = []
    try:
        s = fork_render_settings()
    except Exception as e:
        return [f"could not read fork render settings: {type(e).__name__}: {e}"]

    if s["dpi"] != EXPECTED_DPI:
        problems.append(
            f"render DPI is {s['dpi']}, expected {EXPECTED_DPI} — the fork patch in "
            f"mineru/utils/pdf_image_tools.py was lost. Every page would render at the "
            f"upstream default and VLM table/equation crops would lose ~half their visual "
            f"tokens, silently.")
    if s["page_to_image_defaults"] != EXPECTED_RENDER_DEFAULTS:
        problems.append(
            f"pdf_reader.page_to_image defaults are {s['page_to_image_defaults']}, expected "
            f"{EXPECTED_RENDER_DEFAULTS} — the render-cap patch was lost; 300 DPI would be "
            f"clipped back on A4/Legal.")
    if not str(s["mineru_version"]).startswith("3.4"):
        problems.append(
            f"mineru {s['mineru_version']} is not 3.4.x. 4.0.x DELETES the pipeline backend "
            f"and silently aliases '-b pipeline' to hybrid at forced effort=medium, which "
            f"would remove the dual-backend cross-check without any error.")
    try:
        from mineru.cli.backend_options import LOCAL_BACKEND_CHOICES
        if "pipeline" not in LOCAL_BACKEND_CHOICES:
            problems.append(
                "the 'pipeline' backend is gone from this MinerU — the cross-check that "
                "catches dropped table columns and numeric loss cannot run.")
    except Exception as e:
        problems.append(f"could not read backend choices: {type(e).__name__}: {e}")
    return problems


def make_bbox_salvager(pdf: Path):
    """Return `(fn, close)` where fn(page_idx, bbox) -> the PDF's OWN text inside that box.

    Used to recover KEPT text-family blocks the VLM handed back empty (R8). content_list
    bboxes are normalized to 0-1000 with a TOP-LEFT origin, while pypdfium2 wants PDF
    points with a BOTTOM-LEFT origin — hence the conversion below. Returns (None, noop)
    if the PDF can't be opened, in which case build_markdown simply can't salvage.
    """
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf))
    except Exception as e:
        log(f"  bbox salvage unavailable for {Path(pdf).name}: {type(e).__name__}: {e}")
        return None, (lambda: None)

    def salv(page_idx: int, bbox) -> str:
        page = doc[page_idx]
        W, H = page.get_width(), page.get_height()
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return _lf(page.get_textpage().get_text_bounded(
            left=x0 / 1000 * W, bottom=H - y1 / 1000 * H,
            right=x1 / 1000 * W, top=H - y0 / 1000 * H) or "")

    return salv, doc.close


def make_page_texter(pdf: Path):
    """Return `(fn, close)` where fn(page_idx) -> the whole text layer of that page.
    Needed by the repair pass for captions, whose text usually sits OUTSIDE the figure's
    own bbox."""
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf))
    except Exception:
        return None, (lambda: None)

    def fn(page_idx: int) -> str:
        return _lf(doc[page_idx].get_textpage().get_text_bounded() or "")

    return fn, doc.close


def find_source_pdf(qa: dict[str, Any], output_dir: Path) -> Path | None:
    """Locate the source PDF for an offline rebuild.

    qa.json records the path the PDF had WHEN IT WAS PROCESSED, but the operator convention
    is to move drained PDFs out of `inbox/` into `done/`. A rebuild that silently lost the
    text-layer repairs because the file moved would be a footgun, so also look the file up
    by name in the track's inbox/ and done/ directories."""
    src = qa.get("source_pdf")
    if not src:
        return None
    p = Path(src)
    if p.exists():
        return p
    root = output_dir.parent                      # <track>/output -> <track>
    for sub in ("inbox", "done"):
        for cand in (root / sub).rglob(p.name):
            return cand
    return None


#: MinerU's own client deadline is 3600 s and `plan_tasks` emits ONE task per document
#: for hybrid (mineru/cli/client.py:661-664), so the wall is per-document — a long book
#: fails deterministically while the server is still working correctly, and the client
#: reports that as a parse failure. A 3486 s near-miss (3.2% margin) is already in the
#: logs. local_deploy never set any of these. They are CEILINGS, not waits, so they are
#: generous; PHASE_TIMEOUT_SECONDS below is the real guard. `setdefault` semantics: an
#: operator who exported a value meant it, and the effective values are logged.
BACKEND_TIMEOUT_ENV = {
    "MINERU_TASK_RESULT_TIMEOUT_SECONDS": "86400",           # 24 h per document
    "MINERU_TASK_RESULT_DOWNLOAD_TIMEOUT_SECONDS": "3600",   # result/ZIP retrieval
    "MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS": "900",       # cold MPS model load
    "MINERU_PDF_RENDER_TIMEOUT": "1800",                     # large/complex pages
}

#: Backstop for a genuinely wedged backend, which would otherwise block the cycle
#: forever while holding the run lock. Deliberately far above the measured worst case
#: (7474 s for a batch of 8) so it never fires on slow-but-healthy work.
PHASE_TIMEOUT_SECONDS = int(os.getenv("LOCAL_DEPLOY_PHASE_TIMEOUT_SECONDS", str(12 * 3600)))

RC_TIMEOUT = -9        #: sentinel rc: our own phase timeout fired
RC_SPAWN_FAILED = -1   #: sentinel rc: the backend process could not be started


def build_backend_env(window: int) -> dict[str, str]:
    """The environment a backend subprocess runs under, as a single auditable place.

    Returned rather than applied so the run manifest can record exactly what was in
    effect — MinerU's env vars OVERRIDE its CLI flags (config_reader.py:140-149 returns
    the env value unconditionally when set), so a stale export silently changes a run.
    """
    env = dict(os.environ)
    env["MINERU_MODEL_SOURCE"] = "local"
    # The CLI flag must win over an ambient value: `--window` was previously applied with
    # setdefault, so an exported MINERU_PROCESSING_WINDOW_SIZE silently beat the flag.
    env["MINERU_PROCESSING_WINDOW_SIZE"] = str(window)
    for key, value in BACKEND_TIMEOUT_ENV.items():
        env.setdefault(key, value)
    return env


def run_backend_phase(input_path: Path, out_dir: Path, backend: str, effort: str,
                      window: int, method: str = "auto") -> int:
    """Run one backend over the whole input as a subprocess (models load once, then
    the process exits and frees all memory before the next phase).

    Returns the process rc, or a sentinel (RC_TIMEOUT / RC_SPAWN_FAILED). The caller
    MUST inspect it: a non-zero rc used to be discarded, so a backend that died after
    emitting partial output was indistinguishable from a clean success.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [mineru_bin(), "-p", str(input_path), "-o", str(out_dir),
           "-b", backend, "-f", "true", "-t", "true", "-m", method]
    if backend.startswith("hybrid"):
        cmd += ["--effort", effort, "--image-analysis", "false"]
    env = build_backend_env(window)
    log(f"phase {backend}: {' '.join(cmd)}")
    log(f"  timeouts: phase={PHASE_TIMEOUT_SECONDS}s "
        + " ".join(f"{k.replace('MINERU_', '')}={env[k]}" for k in BACKEND_TIMEOUT_ENV))
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, env=env, timeout=PHASE_TIMEOUT_SECONDS).returncode
    except subprocess.TimeoutExpired:
        rc = RC_TIMEOUT
        log(f"phase {backend} TIMED OUT after {PHASE_TIMEOUT_SECONDS}s and was killed "
            f"(a spawned mineru-api child may survive — check with `pgrep -f mineru`)")
    log(f"phase {backend} finished rc={rc} in {time.time()-t0:.0f}s")
    if rc != 0:
        log(f"  !! {backend} exited non-zero — any output it produced is SUSPECT")
    return rc


def find_output_json(phase_dir: Path, stem: str, kind: str) -> Path | None:
    """Locate <stem>_<kind>.json under a phase output dir, robust to MinerU's nesting
    (OUT/<stem>/auto/<stem>_content_list.json etc.)."""
    hits = glob.glob(str(phase_dir / "**" / f"{stem}*{kind}.json"), recursive=True)
    # prefer an exact basename match; else shortest path (avoids a stale shallow file or a
    # longer-stem sibling shadowing the correct nested output). Excludes content_list_v2
    # for kind='content_list' since '..._v2.json' does not end in 'content_list.json'.
    exact = [h for h in hits if os.path.basename(h) == f"{stem}_{kind}.json"]
    hits = exact or sorted(hits, key=len)
    return Path(hits[0]) if hits else None


def load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_ocr_enabled(middle: Any) -> bool | None:
    """Best-effort read of whether MinerU treated the doc as scanned/OCR (R5)."""
    if not isinstance(middle, dict):
        return None
    for key in ("_ocr_enable", "ocr_enable"):
        if key in middle:
            return bool(middle[key])
    return None


# --- per-paper analysis (offline build) ------------------------------------------

def analyze_paper(pdf: Path, hybrid_cl: list, pipeline_cl: list | None,
                  hybrid_middle: Any) -> dict[str, Any]:
    salv, close_salv = make_bbox_salvager(pdf)
    pgtext, close_pg = make_page_texter(pdf)
    try:
        # R9/R10: fix '??' math-variable corruption and dropped minus signs BEFORE rendering.
        # Both defects are already present in the backend's content_list, so no amount of
        # better rendering can undo them — they must be repaired against the text layer.
        rep = repair.repair_content_list(hybrid_cl, salv, pgtext)
        built = postprocess.build_markdown(hybrid_cl, source_name=pdf.stem, salvage_bbox=salv)
    finally:
        close_salv()
        close_pg()
    built["stats"].update({f"repair_{k}": v for k, v in rep.items()})
    body = built["markdown"]

    # Coverage/cross-check run on the BODY (before the QA header is prepended). Both
    # comparators strip HTML comments, so the header + page markers never affect recall.
    coverage = cov.coverage_report(str(pdf), body, hybrid_cl)
    xcheck = crosscheck.compare(hybrid_cl, pipeline_cl, body) if pipeline_cl is not None else None
    ocr_enabled = detect_ocr_enabled(hybrid_middle)
    pages = max((b.get("page_idx", -1) for b in hybrid_cl
                 if isinstance(b.get("page_idx"), int)), default=-1) + 1

    reasons: list[str] = []      # RELIABLE signals -> needs_review
    notes: list[str] = []        # informational context (NOT a trigger)
    st = built["stats"]
    # 1) Recognition failures (deterministic)
    if st["flagged_tables"]:
        reasons.append(f"{st['flagged_tables']} table(s) failed recognition (flagged in .md)")
    if st["flagged_equations"]:
        reasons.append(f"{st['flagged_equations']} equation(s) failed recognition (flagged in .md)")
    if st["flagged_other"]:
        reasons.append(f"{st['flagged_other']} other flagged block(s)")
    # 2) Body-text loss: word recall + word-based suspect pages (reliable)
    if not coverage["source_reliable"]:
        reasons.append(f"PDF text layer unreliable (PUA {coverage['source_pua_ratio']})")
    else:
        if coverage["word_recall"] < WORD_RECALL_MIN:
            reasons.append(f"word_recall {coverage['word_recall']} < {WORD_RECALL_MIN} "
                           f"(missing words: {coverage['missing_words_sample'][:10]})")
        if coverage["suspect_pages"]:
            reasons.append(f"body text largely absent on pdf page(s) "
                           f"{sorted(p['page_idx'] + 1 for p in coverage['suspect_pages'])} "
                           f"(often figure-only pages — verify)")
    # 3) Numeric loss the deterministic pipeline caught but hybrid lacks
    # R11: the backends disagree on a table's COLUMN COUNT. A recognizer that drops a whole
    # column still emits valid HTML and passes every other guard (VisionTS lost the entire
    # `Informer` column of its headline table, 15 values, undetected), so a second opinion
    # on the shape is the only cheap way to see it.
    if xcheck and xcheck.get("table_shape_divergence"):
        d = xcheck["table_shape_divergence"]
        reasons.append(f"{len(d)} table(s) where the backends disagree on column count "
                       f"(possible dropped/added column): {d[:5]}")
    if xcheck and xcheck["pipeline_vs_hybrid_numeral_recall"] < XCHECK_NUMERAL_MIN:
        reasons.append(f"cross-check: pipeline has >=2-digit numbers absent from hybrid "
                       f"({xcheck['numbers_in_pipeline_not_hybrid'][:10]})")
    # Informational notes (figure/table-confounded or contextual; NOT triggers)
    if coverage["numeral_recall"] < INFO_NUMERAL_RECALL:
        notes.append(f"coverage numeral_recall {coverage['numeral_recall']} is figure/table-confounded "
                     f"(dropped figure axis-labels / glued table digits); numbers to eyeball: "
                     f"{coverage['missing_numbers'][:12]}")
    if xcheck and abs(xcheck["hybrid_table_count"] - xcheck["pipeline_table_count"]) >= 2:
        notes.append(f"table count differs (hybrid {xcheck['hybrid_table_count']} vs "
                     f"pipeline {xcheck['pipeline_table_count']}) — usually pipeline over-splitting a merged table")
    if ocr_enabled:
        notes.append("body text via VLM OCR (robust to bad/rotated text layers; cross-validated vs the text layer)")
    if st.get("salvaged_blocks"):
        notes.append(f"{st['salvaged_blocks']} empty text block(s) ({st['salvaged_chars']} chars) were "
                     f"recovered from the PDF text layer and marked SALVAGED in the .md — the VLM "
                     f"returned nothing for those regions")
    if st.get("empty_blocks_unrecovered"):
        notes.append(f"{st['empty_blocks_unrecovered']} empty text block(s) had no text layer at their "
                     f"bbox (figure region, blank area, or a scanned page) — nothing recoverable")
    notes.append("for number-critical certainty, run an adversarial PDF-vs-Markdown audit (LLM, cell-by-cell)")

    qa = {
        "source_pdf": str(pdf),
        "pages": pages,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "notes": notes,
        "stats": st,
        "dropped_types": built["dropped_types"],
        "flags": built["flags"],
        "coverage": coverage,
        "crosscheck": xcheck,
        "ocr_enabled": ocr_enabled,
    }
    # Fold the trust signal into the .md (agents may never open a sidecar). Header is one
    # HTML-comment block; page markers are inside `body`. Neither reaches the comparators.
    markdown = postprocess.build_qa_header(qa) + "\n\n" + body
    return {"markdown": markdown, "content_list": hybrid_cl, "qa": qa}


def rebuild_markdown(content_list: list, qa: dict | None, output_dir: Path | None = None) -> str:
    """Regenerate a paper's .md from its cached content_list.json (+ qa.json) WITHOUT
    re-running any backend — the .md is a pure function of (body render + QA header).
    Lets `run.py --rebuild` re-emit every .md cheaply when the builder/policy improves,
    instead of paying ~29s/page of VLM inference again."""
    qa = dict(qa) if qa else {}
    if not isinstance(qa.get("pages"), int):   # backfill for caches written before `pages` existed
        qa["pages"] = max((b.get("page_idx", -1) for b in content_list
                           if isinstance(b.get("page_idx"), int)), default=-1) + 1
    # Salvaging empty blocks needs the source PDF; qa.json records its path. If the PDF has
    # moved, rebuild still works — it just can't recover (blocks stay dropped, as before).
    src = find_source_pdf(qa, output_dir) if output_dir else None
    salv, close_salv = (None, (lambda: None))
    pgtext, close_pg = (None, (lambda: None))
    rep = {}
    if src:
        salv, close_salv = make_bbox_salvager(src)
        pgtext, close_pg = make_page_texter(src)
    try:
        if salv is not None:
            rep = repair.repair_content_list(content_list, salv, pgtext)
        built = postprocess.build_markdown(content_list, salvage_bbox=salv)
    finally:
        close_salv()
        close_pg()
    st = built["stats"]
    if rep.get("qq_fixed") or rep.get("signs_fixed"):
        qa.setdefault("stats", {}).update({f"repair_{k}": v for k, v in rep.items()})
        note = (f"{rep['qq_fixed']} corrupted math variable(s) and {rep['signs_fixed']} dropped "
                f"minus sign(s) repaired from the PDF text layer"
                + (f"; {rep['qq_left']} '??' could not be resolved and are left visible"
                   if rep.get("qq_left") else ""))
        notes = [n for n in (qa.get("notes") or []) if not n.startswith("REPAIR:")]
        qa["notes"] = [f"REPAIR: {note}"] + notes
    if st.get("salvaged_blocks"):
        qa.setdefault("stats", {}).update({k: st[k] for k in (
            "salvaged_blocks", "salvaged_chars", "empty_blocks_benign", "empty_blocks_unrecovered")})
        note = (f"{st['salvaged_blocks']} empty text block(s) ({st['salvaged_chars']} chars) recovered "
                f"from the PDF text layer and marked SALVAGED in the .md")
        notes = [n for n in (qa.get("notes") or []) if not n.startswith("SALVAGE:")]
        qa["notes"] = [f"SALVAGE: {note}"] + notes
    return postprocess.build_qa_header(qa) + "\n\n" + built["markdown"]


# --- phases ----------------------------------------------------------------------

def phase_build(pdfs: list[Path], workdir: Path, output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for pdf in pdfs:
        stem = pdf.stem
        hy_cl = load_json(find_output_json(workdir / "hybrid", stem, "content_list"))
        hy_mid = load_json(find_output_json(workdir / "hybrid", stem, "middle"))
        pi_cl = load_json(find_output_json(workdir / "pipeline", stem, "content_list"))
        if hy_cl is None:
            log(f"  !! {stem}: no hybrid content_list found — skipping (hybrid phase failed?)")
            results.append({"stem": stem, "error": "missing hybrid content_list"})
            continue
        res = analyze_paper(pdf, hy_cl, pi_cl, hy_mid)
        sidecar = output / ".sidecar"       # operator-side cache (re-render source + QA record)
        sidecar.mkdir(parents=True, exist_ok=True)
        (output / f"{stem}.md").write_text(res["markdown"], encoding="utf-8")
        (sidecar / f"{stem}.content_list.json").write_text(
            json.dumps(res["content_list"], ensure_ascii=False, indent=1), encoding="utf-8")
        (sidecar / f"{stem}.qa.json").write_text(
            json.dumps(res["qa"], ensure_ascii=False, indent=1), encoding="utf-8")
        verdict = "REVIEW" if res["qa"]["needs_review"] else "ok"
        st = res["qa"]["stats"]
        log(f"  built {stem}.md  [{verdict}]  "
            f"eq={st['equations']} tbl={st['tables']} refs={st['references']} "
            f"cov_num={res['qa']['coverage']['numeral_recall']}")
        results.append({"stem": stem, "qa": res["qa"]})
    return results


def write_report(results: list[dict], output: Path) -> None:
    lines = ["# MinerU batch report", ""]
    review = [r for r in results if r.get("qa", {}).get("needs_review") or r.get("error")]
    lines.append(f"- papers: {len(results)}")
    lines.append(f"- clean: {len(results) - len(review)}")
    lines.append(f"- NEED REVIEW: {len(review)}")
    lines.append("")
    if review:
        lines.append("## Needs human review")
        for r in review:
            if r.get("error"):
                lines.append(f"- **{r['stem']}** — ERROR: {r['error']}")
            else:
                lines.append(f"- **{r['stem']}**")
                for reason in r["qa"]["review_reasons"]:
                    lines.append(f"    - {reason}")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report -> {output/'report.md'}  ({len(review)} need review)")


# --- main ------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Max-precision batch PDF->Markdown (MinerU hybrid+pipeline).")
    ap.add_argument("--input", "-i", required=True, help="PDF file or directory")
    ap.add_argument("--output", "-o", required=True,
                    help="Output directory (agent-facing <stem>.md; JSON cache under .sidecar/)")
    ap.add_argument("--workdir", "-w", default=None, help="Scratch dir for raw backend outputs")
    ap.add_argument("--phases", default="all",
                    help="Comma list of {hybrid,pipeline,build} or 'all'")
    ap.add_argument("--effort", default="high", choices=["medium", "high"])
    ap.add_argument("--method", "-m", default="auto", choices=["auto", "txt", "ocr"],
                    help="hybrid parse method: auto (default, adaptive); txt = character-exact "
                         "native text (born-digital only, no scanned pages); ocr")
    ap.add_argument("--window", type=int, default=32, help="MINERU_PROCESSING_WINDOW_SIZE")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else output / "_work"
    phases = ["hybrid", "pipeline", "build"] if args.phases == "all" else \
        [p.strip() for p in args.phases.split(",") if p.strip()]

    pdfs = find_pdfs(input_path)
    log(f"{len(pdfs)} PDF(s); phases={phases}; workdir={workdir}")

    if {"hybrid", "pipeline"} & set(phases):
        problems = preflight_deps() + preflight_models()
        if problems:
            for p in problems:
                log(f"PREFLIGHT PROBLEM: {p}")
            raise SystemExit("Preflight failed — fix the problem(s) above before running "
                             "(deps: see the pin in local_deploy/README.md; models: "
                             "mineru-models-download -s huggingface -m all)")

    if "hybrid" in phases:
        rc = run_backend_phase(input_path, workdir / "hybrid", "hybrid-engine",
                               args.effort, args.window, args.method)
        if rc != 0:
            log(f"WARNING: hybrid phase exited rc={rc}")
    if "pipeline" in phases:
        rc = run_backend_phase(input_path, workdir / "pipeline", "pipeline",
                               args.effort, args.window, "auto")
        if rc != 0:
            log(f"WARNING: pipeline phase exited rc={rc}")
    if "build" in phases:
        results = phase_build(pdfs, workdir, output)
        write_report(results, output)
        n_review = sum(1 for r in results if r.get("qa", {}).get("needs_review") or r.get("error"))
        log(f"DONE. {len(results)} paper(s), {n_review} need review. Output: {output}")


if __name__ == "__main__":
    main()
