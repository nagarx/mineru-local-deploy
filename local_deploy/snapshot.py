#!/usr/bin/env python3
"""snapshot.py — corpus integrity manifest + offline-rebuild regression baseline.

Two jobs, deliberately in one artifact so they can never drift apart:

  1. INTEGRITY.  SHA-256 of every shipped `.md`, `content_list.json` and `qa.json`.
     `library/` is gitignored, so these files are version-controlled nowhere and a
     move/copy is otherwise unverifiable. This is what proves nothing was lost or
     altered in transit.

  2. REGRESSION BASELINE (Gate A).  SHA-256 of the markdown that `rebuild_markdown`
     produces RIGHT NOW from each cached sidecar. Rebuild is a pure function of
     (content_list, qa, source PDF) and loads no models, so re-running it after a
     refactor and diffing the hashes proves the restructure changed no output.

     The comparison that matters is rebuild-vs-rebuild, NOT rebuild-vs-shipped: the
     636 shipped documents were produced across at least three code eras, so they
     were never a valid baseline for today's code. Measured on this corpus, a
     rebuild-vs-shipped run reports ~13 substantive diffs per 45 documents even with
     zero changes applied — that is the multi-era corpus, not a regression.

Usage:
    .venv/bin/python local_deploy/snapshot.py --library local_deploy/library \\
        --out snapshots/baseline-YYYYMMDD.json [--jobs N]
    .venv/bin/python local_deploy/snapshot.py --compare snapshots/baseline-A.json \\
        --library local_deploy/library
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

SCHEMA_VERSION = "corpus-snapshot-v1"


def sha256_file(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _producer() -> dict:
    """Identify the code that produced this snapshot.

    The git SHA alone is not enough — a dirty tree produces different output from the
    same commit — so the exact bytes of every module that participates in a rebuild are
    hashed too. Borrowed from the unmerged provenance.py on the audit branch.
    """
    rels = ("convert.py", "postprocess.py", "repair.py", "coverage.py", "crosscheck.py")
    h = hashlib.sha256()
    for rel in sorted(rels):
        p = _HERE / rel
        if p.is_file():
            data = p.read_bytes()
            h.update(rel.encode()); h.update(len(data).to_bytes(8, "big")); h.update(data)
    commit = dirty = None
    try:
        commit = subprocess.run(["git", "-C", str(_HERE.parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "-C", str(_HERE.parent), "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import convert
        render = convert.fork_render_settings()
    except Exception:
        render = {}
    return {"git_commit": commit, "git_dirty": dirty,
            "rebuild_code_sha256": h.hexdigest(), "render": render}


def _snapshot_one(args: tuple[str, str]) -> tuple[str, dict]:
    """Worker: hash one document's artifacts and its offline rebuild."""
    track, slug = args
    import convert  # per-process import; keeps the parent light
    out = Path(track) / "output"
    side = out / ".sidecar"
    md, cl_p, qa_p = out / f"{slug}.md", side / f"{slug}.content_list.json", side / f"{slug}.qa.json"
    rec: dict = {
        "track": Path(track).name,
        "md_sha256": sha256_file(md),
        "content_list_sha256": sha256_file(cl_p),
        "qa_sha256": sha256_file(qa_p),
    }
    try:
        cl = convert.load_json(cl_p)
        qa = convert.load_json(qa_p) or {}
        rec["source_pdf_found"] = convert.find_source_pdf(qa, out) is not None
        rec["rebuild_sha256"] = sha256_text(convert.rebuild_markdown(cl, qa, out))
    except Exception as e:
        rec["rebuild_sha256"] = None
        rec["rebuild_error"] = f"{type(e).__name__}: {e}"
    return slug, rec


def build(library: Path, jobs: int) -> dict:
    tasks: list[tuple[str, str]] = []
    for track in sorted(p for p in library.iterdir() if (p / "output").is_dir()):
        for cl in sorted((track / "output" / ".sidecar").glob("*.content_list.json")):
            tasks.append((str(track), cl.name[: -len(".content_list.json")]))
    docs: dict[str, dict] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_snapshot_one, t): t for t in tasks}
        for fut in as_completed(futures):
            slug, rec = fut.result()
            docs[f"{rec['track']}/{slug}"] = rec
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": _producer(),
        "document_count": len(docs),
        "documents": dict(sorted(docs.items())),
    }


def compare(baseline: dict, current: dict) -> int:
    """Report drift. Returns a process exit code."""
    b, c = baseline["documents"], current["documents"]
    only_b, only_c = sorted(set(b) - set(c)), sorted(set(c) - set(b))
    fields = ("md_sha256", "content_list_sha256", "qa_sha256", "rebuild_sha256")
    drift: dict[str, list[str]] = {f: [] for f in fields}
    for k in sorted(set(b) & set(c)):
        for f in fields:
            if b[k].get(f) != c[k].get(f):
                drift[f].append(k)

    print(f"baseline produced by : {baseline['producer'].get('git_commit')} "
          f"(dirty={baseline['producer'].get('git_dirty')})")
    print(f"current  produced by : {current['producer'].get('git_commit')} "
          f"(dirty={current['producer'].get('git_dirty')})")
    print(f"documents: baseline {len(b)}  current {len(c)}")
    if only_b:
        print(f"  MISSING NOW ({len(only_b)}): {only_b[:5]}")
    if only_c:
        print(f"  NEW ({len(only_c)}): {only_c[:5]}")
    print()
    for f in fields:
        n = len(drift[f])
        label = {"md_sha256": "shipped .md (integrity)",
                 "content_list_sha256": "content_list.json (integrity)",
                 "qa_sha256": "qa.json (integrity)",
                 "rebuild_sha256": "offline rebuild (GATE A)"}[f]
        print(f"  {'OK ' if n == 0 else 'DRIFT'}  {label:<34} {n} changed")
        for k in drift[f][:8]:
            print(f"           {k[:76]}")
    bad = len(only_b) + sum(len(v) for v in drift.values())
    print()
    print("SNAPSHOT MATCHES" if bad == 0 else f"{bad} difference(s) — investigate before proceeding")
    return 0 if bad == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="library root holding the track dirs")
    ap.add_argument("--out", help="write a new snapshot here")
    ap.add_argument("--compare", help="compare the live corpus against this snapshot")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="worker processes (default: cores-2)")
    args = ap.parse_args()

    library = Path(args.library).expanduser().resolve()
    if not library.is_dir():
        raise SystemExit(f"--library {library} does not exist")

    print(f"scanning {library} with {args.jobs} worker(s)...", flush=True)
    current = build(library, args.jobs)
    print(f"{current['document_count']} document(s)")

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        raise SystemExit(compare(baseline, current))

    if not args.out:
        raise SystemExit("pass --out to write a snapshot, or --compare to check one")
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(current, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
    rebuilt = sum(1 for d in current["documents"].values() if d.get("rebuild_sha256"))
    found = sum(1 for d in current["documents"].values() if d.get("source_pdf_found"))
    print(f"wrote {out}")
    print(f"  rebuilt OK        : {rebuilt}/{current['document_count']}")
    print(f"  source PDF found  : {found}/{current['document_count']}")


if __name__ == "__main__":
    main()
