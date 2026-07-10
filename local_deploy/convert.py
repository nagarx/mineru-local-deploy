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
    """Verify the pdftext/pypdfium2 pair this MinerU actually works with. pdftext 0.7.x /
    pypdfium2 5.x break the fork twice over: pdf_classify crashes on the removed
    PdfImage.get_pos (silently degrading EVERY doc to forced-OCR) and '-m txt' dies on a
    non-iterable PageChars. A uv re-lock regressed exactly this on 2026-07-08 and cost a
    books run — so refuse to start on a bad pair. The bounds live in pyproject.toml; this
    catches any install that bypassed them."""
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
        try:
            import pypdfium2 as pdfium   # behavior check, not just version strings
            if not hasattr(pdfium.PdfImage, "get_pos"):
                problems.append("pypdfium2.PdfImage.get_pos missing — pdf_classify would silently force-OCR every doc")
        except Exception as e:
            problems.append(f"pypdfium2 import failed: {type(e).__name__}: {e}")
    if problems:
        problems.append('fix: uv pip install "pdftext==0.6.3" "pypdfium2>=4.30,<5"')
    return problems


def run_backend_phase(input_path: Path, out_dir: Path, backend: str, effort: str,
                      window: int, method: str = "auto") -> int:
    """Run one backend over the whole input as a subprocess (models load once, then
    the process exits and frees all memory before the next phase)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [mineru_bin(), "-p", str(input_path), "-o", str(out_dir),
           "-b", backend, "-f", "true", "-t", "true", "-m", method]
    if backend.startswith("hybrid"):
        cmd += ["--effort", effort, "--image-analysis", "false"]
    env = dict(os.environ)
    env["MINERU_MODEL_SOURCE"] = "local"
    env.setdefault("MINERU_PROCESSING_WINDOW_SIZE", str(window))
    log(f"phase {backend}: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    log(f"phase {backend} finished rc={rc} in {time.time()-t0:.0f}s")
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
    built = postprocess.build_markdown(hybrid_cl, source_name=pdf.stem)
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


def rebuild_markdown(content_list: list, qa: dict | None) -> str:
    """Regenerate a paper's .md from its cached content_list.json (+ qa.json) WITHOUT
    re-running any backend — the .md is a pure function of (body render + QA header).
    Lets `run.py --rebuild` re-emit every .md cheaply when the builder/policy improves,
    instead of paying ~29s/page of VLM inference again."""
    qa = dict(qa) if qa else {}
    if not isinstance(qa.get("pages"), int):   # backfill for caches written before `pages` existed
        qa["pages"] = max((b.get("page_idx", -1) for b in content_list
                           if isinstance(b.get("page_idx"), int)), default=-1) + 1
    body = postprocess.build_markdown(content_list)["markdown"]
    return postprocess.build_qa_header(qa) + "\n\n" + body


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
