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
    found: list[str] = []
    for rel in sorted(rels):
        p = _HERE / rel
        if p.is_file():
            data = p.read_bytes()
            h.update(rel.encode()); h.update(len(data).to_bytes(8, "big")); h.update(data)
            found.append(rel)
    if not found:
        # Without this the receipt degrades into sha256(b"") — a stable, meaningless
        # hash that would look like "the code never changed" precisely when the code
        # has MOVED. Fail loudly instead; the module list is maintenance, not magic.
        raise SystemExit(
            f"snapshot.py: none of {list(rels)} were found beside {_HERE}. The "
            f"provenance hash would be the hash of nothing. Update the module list."
        )
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
            "rebuild_code_sha256": h.hexdigest(), "rebuild_code_files": found,
            "render": render}


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


_HASH_FIELDS = (
    ("md_sha256", "shipped .md (integrity)"),
    ("content_list_sha256", "content_list.json (integrity)"),
    ("qa_sha256", "qa.json (integrity)"),
    ("rebuild_sha256", "offline rebuild (GATE A)"),
)

# Render settings that a pure offline rebuild CANNOT observe — rebuild reads cached JSON
# and never rasterises a page. Mutation testing on 2026-08-12 confirmed the blind spot
# exactly: setting EXPECTED_DPI to 72 moved 0 of 200 rebuild hashes. Comparing these here
# is the only thing in this harness that can see a render regression at all.
# `mineru_path` is deliberately NOT an invariant: it is an absolute path and changes
# benignly whenever the tree moves, which it did on 2026-08-11.
_RENDER_INVARIANTS = ("dpi", "page_to_image_defaults", "mineru_version")


def _canon(v):
    """Normalise a value the way a JSON round-trip would.

    A comparison always straddles the JSON boundary: the baseline side was loaded from a
    file (where a tuple has become a list) while the current side is live Python. So
    `fork_render_settings()` returning (300, 4500) and the baseline holding [300, 4500]
    are the SAME setting recorded twice. Without this, the render check false-positives on
    every single run — measured on the first real run of the hardened gate, 2026-08-12.
    """
    if isinstance(v, (tuple, list)):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    return v


def compare(baseline: dict, current: dict, *, allow_new: bool = False) -> int:
    """Report drift between two snapshots. Returns a process exit code.

    Hardened 2026-08-12 (C3). The original had four ways to report success on a corpus
    that had really changed, each verified by reading it rather than assumed:

      1. Documents present only in the CURRENT run were printed but never counted, so a
         gate run after new extractions passed while silently comparing nothing for them.
      2. A field that was None on BOTH sides compared equal, so a document that fails to
         rebuild in both runs — or whose .md is missing in both — read as OK rather than
         as an absence of evidence.
      3. The `producer` block was printed but never compared, so a render-DPI or cap
         regression passed silently (see _RENDER_INVARIANTS).
      4. `source_pdf_found` was recorded and then ignored, so losing PDF resolution —
         which silently disables repair and salvage — was invisible for every document
         that happened to need no repairs.
    """
    fail: list[str] = []
    note: list[str] = []

    bs, cs = baseline.get("schema_version"), current.get("schema_version")
    if bs != cs:
        print(f"SCHEMA MISMATCH: baseline {bs!r} vs current {cs!r} — not comparable.")
        return 2

    bp, cp = baseline.get("producer") or {}, current.get("producer") or {}
    print(f"baseline produced by : {bp.get('git_commit')} (dirty={bp.get('git_dirty')})")
    print(f"current  produced by : {cp.get('git_commit')} (dirty={cp.get('git_dirty')})")

    if bp.get("git_dirty"):
        note.append("baseline came from a DIRTY tree — it cannot be reproduced from its "
                    "commit. Re-baseline from a clean tree.")
    if bp.get("rebuild_code_sha256") == cp.get("rebuild_code_sha256"):
        note.append("rebuild code is byte-identical to the baseline (so identical output "
                    "proves nothing about a refactor — there was none).")
    else:
        note.append("rebuild code CHANGED since the baseline — this is what the gate is "
                    "here to measure.")
    if (bf := bp.get("rebuild_code_files")) != (cf := cp.get("rebuild_code_files")):
        note.append(f"modules hashed: baseline {bf or '<not recorded>'} -> current {cf}")

    # --- render invariants: the one regression class rebuild-vs-rebuild cannot see ---
    br, cr = bp.get("render") or {}, cp.get("render") or {}
    for k in _RENDER_INVARIANTS:
        if _canon(br.get(k)) != _canon(cr.get(k)):
            fail.append(f"render.{k}: {br.get(k)!r} -> {cr.get(k)!r}")

    # --- document population ---------------------------------------------------------
    b, c = baseline["documents"], current["documents"]
    only_b, only_c = sorted(set(b) - set(c)), sorted(set(c) - set(b))
    print(f"documents: baseline {len(b)}  current {len(c)}")
    if only_b:
        print(f"  MISSING NOW ({len(only_b)}): {only_b[:5]}")
        fail.append(f"{len(only_b)} document(s) present in the baseline are gone")
    if only_c:
        print(f"  NEW ({len(only_c)}): {only_c[:5]}")
        if allow_new:
            note.append(f"{len(only_c)} new document(s) — not compared (--allow-new)")
        else:
            fail.append(f"{len(only_c)} document(s) are new and therefore uncompared "
                        f"(pass --allow-new if that is intended)")

    # --- per-field drift, with absence treated as absence of evidence ----------------
    shared = sorted(set(b) & set(c))
    print()
    for f, label in _HASH_FIELDS:
        drift = [k for k in shared if b[k].get(f) != c[k].get(f)]
        # None on the CURRENT side means the artifact is missing or the rebuild threw.
        # The old code let None == None pass as agreement; it is not agreement.
        unusable = [k for k in shared if c[k].get(f) is None]
        state = "OK   " if not drift and not unusable else "DRIFT"
        extra = f", {len(unusable)} unusable" if unusable else ""
        print(f"  {state}  {label:<34} {len(drift)} changed{extra}")
        for k in drift[:8]:
            print(f"           {k[:76]}")
        for k in unusable[:5]:
            print(f"           [no value] {k[:66]}")
        if drift:
            fail.append(f"{label}: {len(drift)} changed")
        if unusable:
            fail.append(f"{label}: {len(unusable)} document(s) produced no value")

    # --- state fields that were recorded and never checked ---------------------------
    lost_pdf = [k for k in shared
                if b[k].get("source_pdf_found") and not c[k].get("source_pdf_found")]
    gained_pdf = [k for k in shared
                  if c[k].get("source_pdf_found") and not b[k].get("source_pdf_found")]
    new_err = [k for k in shared if c[k].get("rebuild_error") and not b[k].get("rebuild_error")]
    print(f"  {'OK   ' if not lost_pdf else 'DRIFT'}  {'source PDF resolution':<34} "
          f"{len(lost_pdf)} lost, {len(gained_pdf)} gained")
    if lost_pdf:
        # Losing the PDF silently disables repair and salvage. For a document that needed
        # neither, the rebuild hash is unchanged — so this is invisible to every other row.
        fail.append(f"source PDF resolution: {len(lost_pdf)} document(s) lost it "
                    f"(repair and salvage go silently dead)")
        for k in lost_pdf[:5]:
            print(f"           {k[:76]}")
    if gained_pdf:
        note.append(f"{len(gained_pdf)} document(s) newly resolve their source PDF")
    if new_err:
        fail.append(f"rebuild raised for {len(new_err)} document(s) that were fine before")
        print(f"  DRIFT  {'rebuild exceptions':<34} {len(new_err)} new")
        for k in new_err[:5]:
            print(f"           {c[k].get('rebuild_error', '')[:60]}  {k[:40]}")

    print()
    for n in note:
        print(f"  note: {n}")
    print()
    if fail:
        print(f"GATE FAILED — {len(fail)} finding(s):")
        for f_ in fail:
            print(f"  - {f_}")
        return 1
    print("SNAPSHOT MATCHES")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="library root holding the track dirs")
    ap.add_argument("--out", help="write a new snapshot here")
    ap.add_argument("--compare", help="compare the live corpus against this snapshot")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="worker processes (default: cores-2)")
    ap.add_argument("--allow-new", action="store_true",
                    help="with --compare: tolerate documents absent from the baseline. "
                         "They cannot be checked, so this weakens the gate — use it only "
                         "when new extractions between the two runs are expected.")
    args = ap.parse_args()

    library = Path(args.library).expanduser().resolve()
    if not library.is_dir():
        raise SystemExit(f"--library {library} does not exist")

    print(f"scanning {library} with {args.jobs} worker(s)...", flush=True)
    current = build(library, args.jobs)
    print(f"{current['document_count']} document(s)")

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        raise SystemExit(compare(baseline, current, allow_new=args.allow_new))

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
