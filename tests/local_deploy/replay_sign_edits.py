#!/usr/bin/env python3
"""Corpus acceptance check for repair.py's sign pass.

`repair_signs` is the only code that MUTATES extracted content, and it has shipped
mathematical corruption twice. Unit goldens pin the known cases; this replays the pass
over the WHOLE corpus and reports every edit it would make today, so a change in its
behaviour is visible as a number rather than discovered in a deliverable months later.

It reads the cached content_list + the real source PDF and never writes to the corpus.

Usage:
    .venv/bin/python vendor/mineru/tests/local_deploy/replay_sign_edits.py \\
        --corpora ~/code_local/minerU_pipeline_data/corpora [--jobs N] [--max-edits N]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_CODE = Path(__file__).resolve().parents[2] / "local_deploy"
sys.path.insert(0, str(_CODE))


def _replay_one(args: tuple[str, str]) -> dict:
    track, slug = args
    import convert
    import repair
    out = Path(track) / "output"
    side = out / ".sidecar"
    cl = convert.load_json(side / f"{slug}.content_list.json")
    qa = convert.load_json(side / f"{slug}.qa.json") or {}
    if cl is None:
        return {"slug": slug, "track": Path(track).name, "error": "no content_list"}
    pdf = convert.find_source_pdf(qa, out)
    if pdf is None:
        return {"slug": slug, "track": Path(track).name, "nopdf": True}
    salv, close_s = convert.make_bbox_salvager(pdf)
    pg, close_p = convert.make_page_texter(pdf)
    try:
        st = repair.repair_content_list(cl, salv, pg)
    except Exception as e:                                    # noqa: BLE001
        return {"slug": slug, "track": Path(track).name, "error": f"{type(e).__name__}: {e}"}
    finally:
        close_s()
        close_p()
    edits = [e for e in st.get("sign_edits", []) if e.get("action") == "signed"]
    return {
        "slug": slug,
        "track": Path(track).name,
        "signed": st.get("signs_fixed", 0),
        "declined": st.get("signs_declined", 0),
        "qq_fixed": st.get("qq_fixed", 0),
        "contexts": [e.get("context", "")[:120] for e in edits],
    }


def classify(ctx: str) -> str:
    """Bucket an edit by the syntactic role of what precedes the signed number."""
    head = ctx[:ctx.rfind(" ")] if " " in ctx else ctx
    if re.search(r"\\(frac|dfrac|tfrac|sqrt)\s*\{[^}]*$", head):
        return "frac/sqrt argument"
    if re.search(r"[\^_]\s*\{?[^}]*$", head[-8:]):
        return "sup/subscript"
    if re.search(r"[+\-*/=(\[{]\s*$", head):
        return "after operator/bracket"
    if re.search(r"[A-Za-z]\s*$", head):
        return "after letter/word"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-edits", type=int, default=None,
                    help="exit non-zero if more than this many sign edits are made")
    ap.add_argument("--out", help="write the full per-document record here")
    args = ap.parse_args()

    root = Path(args.corpora).expanduser().resolve()
    tasks = []
    for track in sorted(p for p in root.iterdir() if (p / "output" / ".sidecar").is_dir()):
        for f in sorted(glob.glob(str(track / "output" / ".sidecar" / "*.content_list.json"))):
            tasks.append((str(track), os.path.basename(f)[: -len(".content_list.json")]))

    print(f"replaying the sign pass over {len(tasks)} document(s), {args.jobs} worker(s)...",
          flush=True)
    recs, done = [], 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for fut in as_completed([ex.submit(_replay_one, t) for t in tasks]):
            recs.append(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    edited = [r for r in recs if r.get("signed")]
    total = sum(r.get("signed", 0) for r in recs)
    declined = sum(r.get("declined", 0) for r in recs)
    nopdf = sum(1 for r in recs if r.get("nopdf"))
    errs = [r for r in recs if r.get("error")]
    buckets = collections.Counter()
    for r in edited:
        for c in r.get("contexts", []):
            buckets[classify(c)] += 1

    print()
    print(f"documents         : {len(recs)}  (no PDF: {nopdf}, errors: {len(errs)})")
    print(f"documents edited  : {len(edited)}")
    print(f"sign edits applied: {total}")
    print(f"candidates declined: {declined}")
    if buckets:
        print("\nedits by preceding context (anything but prose is suspect):")
        for k, v in buckets.most_common():
            print(f"  {v:>5}  {k}")
    if edited:
        print("\nsample edits:")
        for r in edited[:8]:
            for c in r["contexts"][:1]:
                print(f"  [{r['track']}] {r['slug'][:44]}: {c!r}")
    for e in errs[:5]:
        print(f"  ERROR {e['slug'][:50]}: {e['error']}")

    if args.out:
        Path(args.out).write_text(json.dumps(recs, indent=1, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.max_edits is not None and total > args.max_edits:
        raise SystemExit(f"FAIL: {total} sign edits exceeds --max-edits {args.max_edits}")


if __name__ == "__main__":
    main()
