#!/usr/bin/env python3
"""
run.py — cyclic, resumable, never-miss driver for a large PDF library.

Drop PDFs into  <root>/inbox/.  --root is REQUIRED — this repo runs two independent tracks,
research_papers/ and books/, each a self-contained library (own inbox/output/ledger) so they
never mix. Each invocation processes ONE cycle of the next N pending PDFs and exits; run it as
often as you like, or pass --loop to drain the whole inbox unattended. State lives in an SQLite
ledger, so it is fully resumable — kill it anytime and re-run; nothing is reprocessed or missed.

  <root>/               (research_papers/ or books/ — pick exactly one per invocation)
    inbox/            you drop PDFs here (recursive; subfolders fine)
    output/           <slug>.md   <- the ONLY agent-facing file (QA header + page markers inside)
      .sidecar/       <slug>.content_list.json + <slug>.qa.json   <- operator cache, NOT for agents
    work/             scratch (staging symlinks + backend outputs); cleaned each cycle
    state/ledger.db   the ledger (source of truth) + run.lock (single-instance guard)
    report.md         rolling status

Usage (pick a track with --root — research_papers or books):
  R=local_deploy/library/research_papers               # (or .../library/books for the books track)
  python local_deploy/run.py --root $R --loop          # keep cycling until the inbox is drained
  python local_deploy/run.py --root $R                 # one cycle of the next --batch PDFs
  python local_deploy/run.py --root $R --status        # print counts + refresh report.md
  python local_deploy/run.py --root $R --verify        # assert every 'done' paper has its .md
  python local_deploy/run.py --root $R --retry-failed  # requeue failed papers, then cycle
  python local_deploy/run.py --root $R --rebuild       # re-emit every .md from the .sidecar cache (offline, no models)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import convert   # noqa: E402  (validated per-paper engine)
import library   # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")   # foo.content_list.json -> ...json.tmp (last ext only)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)                              # atomic on POSIX; a kill never leaves a half file


def _pages_of(content_list: list) -> int:
    return max((b.get("page_idx", -1) for b in content_list), default=-1) + 1


def _stage_and_run(work: Path, items: list[dict], cfg: dict) -> tuple[Path, Path, bool]:
    """Symlink items as <id>.pdf into a fresh staging dir and run both backends over it.
    Returns (hy_dir, pi_dir, spawn_ok). spawn_ok=False means the mineru process couldn't
    even start (missing binary / OS spawn error) — never propagates."""
    staging, hy_dir, pi_dir = work / "staging", work / "hybrid", work / "pipeline"
    for d in (staging, hy_dir, pi_dir):
        shutil.rmtree(d, ignore_errors=True)
    staging.mkdir(parents=True)
    for it in items:
        (staging / f"{it['id']}.pdf").symlink_to(Path(it["source_path"]).resolve())
    spawn_ok = True
    try:
        convert.run_backend_phase(staging, hy_dir, "hybrid-engine", cfg["effort"], cfg["window"], cfg["method"])
        convert.run_backend_phase(staging, pi_dir, "pipeline", cfg["effort"], cfg["window"], "auto")
    except Exception as e:   # OS couldn't spawn mineru (missing binary, etc.)
        log(f"  backend could not start: {type(e).__name__}: {e}")
        spawn_ok = False
    shutil.rmtree(staging, ignore_errors=True)
    return hy_dir, pi_dir, spawn_ok


def _build_paper(lib: library.Library, output: Path, hy_dir: Path, pi_dir: Path,
                 fid: str, it: dict) -> str:
    """Build one paper from existing backend outputs. Returns 'done' | 'no_output' | 'failed'.
    'no_output' is NOT marked failed here — the caller decides (isolation vs genuine)."""
    hy_cl = convert.load_json(convert.find_output_json(hy_dir, fid, "content_list"))
    if hy_cl is None:
        return "no_output"
    try:
        hy_mid = convert.load_json(convert.find_output_json(hy_dir, fid, "middle"))
        pi_cl = convert.load_json(convert.find_output_json(pi_dir, fid, "content_list"))
        res = convert.analyze_paper(Path(it["source_path"]), hy_cl, pi_cl, hy_mid)
        slug = it["slug"]
        md_path = output / f"{slug}.md"
        sidecar = output / ".sidecar"       # operator cache (re-render source + QA record), out of the agent path
        sidecar.mkdir(parents=True, exist_ok=True)
        _atomic_write(md_path, res["markdown"])
        _atomic_write(sidecar / f"{slug}.content_list.json",
                      json.dumps(res["content_list"], ensure_ascii=False, indent=1))
        _atomic_write(sidecar / f"{slug}.qa.json",
                      json.dumps(res["qa"], ensure_ascii=False, indent=1))
        lib.mark_done(fid, str(md_path), res["qa"]["needs_review"], pages=_pages_of(hy_cl))
        log(f"  done {slug}  [{'REVIEW' if res['qa']['needs_review'] else 'ok'}]  "
            f"pages={_pages_of(hy_cl)} xcheck={'yes' if pi_cl is not None else 'no-pipeline'}")
        return "done"
    except Exception as e:
        lib.mark_failed(fid, f"build error: {type(e).__name__}: {e}")
        log(f"  FAILED {it['slug']}: build error {type(e).__name__}: {e}")
        return "failed"


def rebuild_all(output: Path) -> int:
    """Re-emit every output/<slug>.md from the .sidecar cache — no backend, no models.
    Applies builder/policy changes (e.g. a better table renderer) across the whole library
    without paying VLM inference again. Also migrates any legacy root-level sidecars
    (written before the .sidecar split) into .sidecar/ first."""
    sidecar = output / ".sidecar"
    sidecar.mkdir(parents=True, exist_ok=True)
    for f in list(output.glob("*.content_list.json")) + list(output.glob("*.qa.json")):
        f.replace(sidecar / f.name)          # migrate legacy layout into the cache dir
    n = 0
    for clf in sorted(sidecar.glob("*.content_list.json")):
        slug = clf.name[: -len(".content_list.json")]
        cl = convert.load_json(clf)
        if cl is None:
            log(f"  skip {slug}: unreadable content_list cache")
            continue
        qa = convert.load_json(sidecar / f"{slug}.qa.json")
        _atomic_write(output / f"{slug}.md", convert.rebuild_markdown(cl, qa))
        n += 1
    return n


def process_cycle(lib: library.Library, root: Path, batch_size: int, cfg: dict) -> int:
    """Run one cycle. Returns number of PDFs attempted (0 = nothing pending)."""
    output, work = root / "output", root / "work"
    output.mkdir(parents=True, exist_ok=True)

    new = lib.register_inbox(root / "inbox")
    if new:
        log(f"registered {new} new PDF(s) from inbox")

    batch = lib.next_batch(batch_size)
    if not batch:
        return 0

    # drop items whose source vanished between registration and now (retryable)
    items = []
    for it in batch:
        if Path(it["source_path"]).exists():
            items.append(it)
        else:
            lib.mark_failed(it["id"], f"source file missing: {it['source_path']}")
            log(f"  FAILED {it['slug']}: source missing")
    if not items:
        return len(batch)

    log(f"cycle: processing {len(items)} PDF(s)")
    hy_dir, pi_dir, spawn_ok = _stage_and_run(work, items, cfg)

    no_output: list[dict] = []
    for it in items:
        status = _build_paper(lib, output, hy_dir, pi_dir, it["id"], it)
        if status == "no_output":
            no_output.append(it)

    # Poison isolation: if some papers produced no backend output but the process COULD
    # spawn, re-run each alone so a single bad PDF doesn't take its batch-mates down.
    if no_output:
        if spawn_ok and len(items) > 1:
            log(f"{len(no_output)} paper(s) had no backend output — re-running each in "
                f"isolation to spare their batch-mates")
            for it in no_output:
                h2, p2, ok2 = _stage_and_run(work, [it], cfg)
                if _build_paper(lib, output, h2, p2, it["id"], it) == "no_output":
                    lib.mark_failed(it["id"], "no content_list even in isolation "
                                              "(unparseable/poison PDF or backend crash)")
                    log(f"  FAILED {it['slug']}: no output in isolation")
        else:
            for it in no_output:
                lib.mark_failed(it["id"], "backend produced no content_list")
                log(f"  FAILED {it['slug']}: no backend output")

    shutil.rmtree(work / "hybrid", ignore_errors=True)
    shutil.rmtree(work / "pipeline", ignore_errors=True)
    return len(batch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cyclic, resumable PDF->Markdown library driver.")
    ap.add_argument("--root", required=True,
                    help="library root for the track to run (REQUIRED; no default, so you never "
                         "run the wrong track): e.g. local_deploy/library/research_papers or "
                         ".../library/books")
    ap.add_argument("--batch", type=int, default=8, help="PDFs per cycle (default 8)")
    ap.add_argument("--loop", action="store_true", help="keep cycling until the inbox is drained")
    ap.add_argument("--status", action="store_true", help="print counts + refresh report.md, then exit")
    ap.add_argument("--verify", action="store_true", help="assert every 'done' paper has its .md, then exit")
    ap.add_argument("--retry-failed", action="store_true", help="requeue all failed papers before cycling")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-emit every output/*.md from the .sidecar cache (offline, no models); "
                         "applies builder/policy changes without re-running any backend")
    ap.add_argument("--effort", default="high", choices=["medium", "high"])
    ap.add_argument("--method", default="auto", choices=["auto", "txt", "ocr"])
    ap.add_argument("--window", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():   # a typo'd --root must not silently create a fresh empty track
        raise SystemExit(f"--root {root} does not exist — mkdir it first if you meant to start a new track")
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    lib = library.Library(root / "state" / "ledger.db")

    if args.status:
        lib.register_inbox(root / "inbox")
        lib.write_report(root / "report.md")
        print(json.dumps(lib.counts(), indent=2))
        print(f"report: {root / 'report.md'}")
        return
    if args.verify:
        missing = lib.verify_outputs()
        print("MISSING outputs for 'done' papers:", missing or "none — all accounted for")
        print(json.dumps(lib.counts(), indent=2))
        return

    # single-instance guard: two concurrent runs on one root would double-process.
    lock_fh = open(root / "state" / "run.lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run.py is already active on {root} (state/run.lock held)")

    if args.rebuild:   # offline re-render from cache — no models needed, but hold the lock
        n = rebuild_all(root / "output")
        log(f"rebuilt {n} markdown file(s) from the .sidecar cache -> {root / 'output'}")
        lib.close()
        return

    problems = convert.preflight_deps() + convert.preflight_models()
    if problems:
        for p in problems:
            log(f"PREFLIGHT PROBLEM: {p}")
        raise SystemExit("Preflight failed — fix the problem(s) above before running "
                         "(deps: see the pin in local_deploy/README.md; models: "
                         "mineru-models-download -s huggingface -m all)")

    if args.retry_failed:
        log(f"requeued {lib.requeue_failed()} failed paper(s)")

    cfg = {"effort": args.effort, "method": args.method, "window": args.window}
    log(f"library: {root}  | batch={args.batch} effort={args.effort} method={args.method}")
    while True:
        stuck = lib.reset_stuck()   # recover any in_progress stranded by a prior cycle/run
        if stuck:
            log(f"recovered {stuck} stranded in-progress paper(s)")
        n = process_cycle(lib, root, args.batch, cfg)
        lib.write_report(root / "report.md")
        c = lib.counts()
        log(f"status: done {c['done']}/{c['total']} | pending {c['pending']} | "
            f"failed {c['failed']} | review {c['needs_review']}")
        if n == 0:
            log("inbox drained — library up to date")
            break
        if not args.loop:
            break
    log(f"report: {root / 'report.md'}")
    lib.close()


if __name__ == "__main__":
    main()
