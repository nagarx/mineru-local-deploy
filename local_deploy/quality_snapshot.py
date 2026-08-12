#!/usr/bin/env python3
"""quality_snapshot.py — Gate A2: a regression baseline for the QUALITY modules.

WHY THIS EXISTS. Gate A (snapshot.py) rebuilds every document and hash-compares, which
sounds total but is not: `rebuild_markdown` calls only `repair` and `postprocess`. It
never calls `coverage` or `crosscheck`, and never evaluates the review gate. Mutation
testing over 200 real documents on 2026-08-12 measured the consequence exactly —
decapitating `coverage_report`, decapitating `crosscheck.compare`, and gutting
WORD_RECALL_MIN / XCHECK_NUMERAL_MIN to 0.10 each produced **0 detected changes**.

So Gate A alone cannot license a refactor of those modules, and the W3 restructure splits
`coverage.py` into `quality/textmetrics.py` + `quality/recall.py` — precisely the code it
cannot see. This harness closes that hole the same way Gate A closed the render path:
characterise the real functions over the real corpus, hash the results, and diff.

WHAT IT PINS, per document:
  coverage   sha256 of coverage_report(pdf, md, content_list)   [needs the source PDF]
  xtext      sha256 of crosscheck.content_list_plain_text(cl)
  xcompare   sha256 of crosscheck.compare(cl, PERTURB(cl), md)
  gate       the review-gate decision recomputed from the coverage/crosscheck output

The perturbation is deterministic and defined HERE, not in the library, so it cannot
drift with the code under test. The real second-opinion content_list is not available —
`work/` is deleted after each cycle — so a synthetic one is the only way to drive
`crosscheck.compare` through its branches at all. It does not need to be realistic; it
needs to be STABLE and to exercise numeral recall, type counts and table shape.

Passing the whole shipped .md as `markdown` is deliberate and safe: both comparators
strip `<!--...-->` before anything else, so the QA header cannot affect the result.

Usage:
    .venv/bin/python local_deploy/quality_snapshot.py --library <corpora> \\
        --out snapshots/quality-YYYYMMDD.json [--jobs N]
    .venv/bin/python local_deploy/quality_snapshot.py --library <corpora> \\
        --compare snapshots/quality-YYYYMMDD.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

SCHEMA_VERSION = "quality-snapshot-v1"

# Review-gate thresholds are recomputed here rather than imported, so that a refactor
# which silently changes a threshold shows up as a DIFF instead of being adopted as the
# new truth. Mutation testing proved this matters: gutting both to 0.10 was invisible.
WORD_RECALL_MIN = 0.90
XCHECK_NUMERAL_MIN = 0.90
INFO_NUMERAL_RECALL = 0.97

_LAST_CELL = re.compile(r"<t([dh])\b[^>]*>(?:(?!</t\1>).)*</t\1>\s*(?=</tr>)", re.S | re.I)
_DIGITS = re.compile(r"\d")


def _canon_sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _perturb(content_list: list[dict]) -> list[dict]:
    """A deterministic stand-in for the pipeline backend's second opinion.

    Three edits, each aimed at one branch of crosscheck.compare:
      - drop every 7th block            -> block type counts diverge
      - strip digits from every 5th text block -> numeral recall drops
      - delete the last cell of every row of the first table -> shape divergence
    Deliberately crude. Its only jobs are to be stable across runs and to make sure the
    comparison code actually executes rather than short-circuiting on identical inputs.
    """
    out: list[dict] = []
    table_done = False
    for i, blk in enumerate(content_list):
        if i % 7 == 6:
            continue
        b = copy.deepcopy(blk)
        if i % 5 == 0 and isinstance(b.get("text"), str):
            b["text"] = _DIGITS.sub("", b["text"])
        if not table_done and isinstance(b.get("table_body"), str) and "<tr" in b["table_body"].lower():
            b["table_body"] = _LAST_CELL.sub("", b["table_body"])
            table_done = True
        out.append(b)
    return out


def _review_gate(coverage: dict | None, xcheck: dict | None) -> dict:
    """The review-gate decision, recomputed from comparator output.

    This mirrors the `reasons` / `notes` logic of convert.analyze_paper. It is duplicated
    on purpose — see the note on the thresholds above. What is pinned is the DECISION
    SHAPE (which conditions fired), not the prose, so that rewording a message does not
    fail the gate but changing a threshold or an operator does.
    """
    fired: list[str] = []
    if coverage is None:
        return {"fired": ["no_coverage"], "needs_review": None}
    if not coverage["source_reliable"]:
        fired.append("source_unreliable")
    else:
        if coverage["word_recall"] < WORD_RECALL_MIN:
            fired.append("word_recall_below_min")
        if coverage["suspect_pages"]:
            fired.append(f"suspect_pages:{len(coverage['suspect_pages'])}")
    if xcheck:
        if xcheck.get("table_shape_divergence"):
            fired.append(f"table_shape_divergence:{len(xcheck['table_shape_divergence'])}")
        if xcheck["pipeline_vs_hybrid_numeral_recall"] < XCHECK_NUMERAL_MIN:
            fired.append("xcheck_numeral_below_min")
    notes: list[str] = []
    if coverage["numeral_recall"] < INFO_NUMERAL_RECALL:
        notes.append("numeral_recall_informational")
    return {"fired": fired, "notes": notes, "needs_review": bool(fired)}


def _one(args: tuple[str, str]) -> tuple[str, dict]:
    track, slug = args
    import convert
    import coverage as cov
    import crosscheck

    out = Path(track) / "output"
    side = out / ".sidecar"
    rec: dict = {"track": Path(track).name}
    try:
        cl = convert.load_json(side / f"{slug}.content_list.json")
        qa = convert.load_json(side / f"{slug}.qa.json") or {}
        md = (out / f"{slug}.md").read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return slug, {**rec, "error": f"load: {type(e).__name__}: {e}"}

    coverage_out = None
    try:
        pdf = convert.find_source_pdf(qa, out)
        rec["source_pdf_found"] = pdf is not None
        if pdf is not None:
            coverage_out = cov.coverage_report(str(pdf), md, cl)
            rec["coverage"] = _canon_sha(coverage_out)
            # a couple of headline scalars in the clear, so a diff is readable without
            # re-running anything
            rec["word_recall"] = coverage_out["word_recall"]
            rec["numeral_recall"] = coverage_out["numeral_recall"]
    except Exception as e:
        rec["coverage_error"] = f"{type(e).__name__}: {e}"

    xcheck_out = None
    try:
        rec["xtext"] = _canon_sha(crosscheck.content_list_plain_text(cl))
        # ABSOLUTE, not differential. `table_shape_divergence` compares widths(hybrid) vs
        # widths(pipeline), so any change to the width PARSER shifts both sides equally and
        # cancels. Mutation testing on 2026-08-12 demonstrated it: breaking the colspan
        # regex moved 0 of 120 documents through `xcompare` alone, even though 283 of the
        # 636 documents contain colspan. Pinning the widths themselves is what sees it.
        rec["xwidths"] = _canon_sha(crosscheck._table_widths(cl))
        xcheck_out = crosscheck.compare(cl, _perturb(cl), md)
        rec["xcompare"] = _canon_sha(xcheck_out)
    except Exception as e:
        rec["xcheck_error"] = f"{type(e).__name__}: {e}"

    rec["gate"] = _canon_sha(_review_gate(coverage_out, xcheck_out))
    return slug, rec


def _producer() -> dict:
    """Hash the bytes of the modules this gate actually exercises."""
    rels = ("coverage.py", "crosscheck.py", "convert.py")
    h = hashlib.sha256()
    found: list[str] = []
    for rel in sorted(rels):
        p = _HERE / rel
        if p.is_file():
            data = p.read_bytes()
            h.update(rel.encode()); h.update(len(data).to_bytes(8, "big")); h.update(data)
            found.append(rel)
    if not found:
        raise SystemExit(f"quality_snapshot.py: none of {list(rels)} found beside {_HERE}; "
                         f"the provenance hash would be the hash of nothing.")
    # BOTH copies are recorded, and compare() fails when they diverge. The pinned values
    # above are this gate's own; `live` is what the pipeline will actually apply. Recording
    # only the pinned pair looked like a threshold check but was not one — mutation testing
    # on 2026-08-12 showed a gutted production threshold sailing through, because the
    # thresholds are also invisible in the per-document hashes (only 3 of 636 documents sit
    # below word_recall 0.90, so moving the line changes no decision).
    live: dict = {}
    try:
        import convert
        for k in ("WORD_RECALL_MIN", "XCHECK_NUMERAL_MIN", "INFO_NUMERAL_RECALL"):
            live[k] = getattr(convert, k, "<absent>")
    except Exception as e:
        live = {"<import failed>": f"{type(e).__name__}: {e}"}
    return {"quality_code_sha256": h.hexdigest(), "quality_code_files": found,
            "thresholds": {"WORD_RECALL_MIN": WORD_RECALL_MIN,
                           "XCHECK_NUMERAL_MIN": XCHECK_NUMERAL_MIN,
                           "INFO_NUMERAL_RECALL": INFO_NUMERAL_RECALL},
            "thresholds_live": live}


def build(library: Path, jobs: int) -> dict:
    tasks: list[tuple[str, str]] = []
    for track in sorted(p for p in library.iterdir() if (p / "output").is_dir()):
        for cl in sorted((track / "output" / ".sidecar").glob("*.content_list.json")):
            tasks.append((str(track), cl.name[: -len(".content_list.json")]))
    docs: dict[str, dict] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_one, t): t for t in tasks}
        for fut in as_completed(futures):
            slug, rec = fut.result()
            docs[f"{rec['track']}/{slug}"] = rec
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}", flush=True)
    return {"schema_version": SCHEMA_VERSION, "producer": _producer(),
            "document_count": len(docs), "documents": dict(sorted(docs.items()))}


_FIELDS = (("coverage", "coverage_report()"),
           ("xtext", "crosscheck plain text"),
           ("xwidths", "table widths (absolute)"),
           ("xcompare", "crosscheck.compare()"),
           ("gate", "review-gate DECISION"))


def compare(baseline: dict, current: dict) -> int:
    if baseline.get("schema_version") != current.get("schema_version"):
        print("SCHEMA MISMATCH — not comparable.")
        return 2
    bp, cp = baseline.get("producer") or {}, current.get("producer") or {}
    b, c = baseline["documents"], current["documents"]
    fail: list[str] = []

    if bp.get("thresholds") != cp.get("thresholds"):
        fail.append(f"review-gate thresholds changed: {bp.get('thresholds')} -> "
                    f"{cp.get('thresholds')}")
    # The production thresholds must match this gate's pinned copy. They are nearly
    # invisible in the per-document hashes -- only 3 of 636 documents sit below
    # word_recall 0.90 -- so without this a gutted threshold passes silently.
    if cp.get("thresholds_live") is not None and cp.get("thresholds_live") != cp.get("thresholds"):
        fail.append(f"LIVE thresholds in convert.py do not match the pinned ones: "
                    f"{cp.get('thresholds_live')} vs {cp.get('thresholds')}")
    if bp.get("thresholds_live") != cp.get("thresholds_live"):
        fail.append(f"live thresholds changed: {bp.get('thresholds_live')} -> "
                    f"{cp.get('thresholds_live')}")
    print(f"quality code: {'UNCHANGED' if bp.get('quality_code_sha256') == cp.get('quality_code_sha256') else 'CHANGED'}"
          f"  files={cp.get('quality_code_files')}")
    print(f"documents: baseline {len(b)}  current {len(c)}")

    only_b, only_c = sorted(set(b) - set(c)), sorted(set(c) - set(b))
    if only_b:
        fail.append(f"{len(only_b)} document(s) vanished")
    if only_c:
        fail.append(f"{len(only_c)} document(s) are new and uncompared")
    shared = sorted(set(b) & set(c))
    print()
    for f, label in _FIELDS:
        drift = [k for k in shared if b[k].get(f) != c[k].get(f)]
        missing = [k for k in shared if c[k].get(f) is None and b[k].get(f) is not None]
        print(f"  {'OK   ' if not drift and not missing else 'DRIFT'}  {label:<28} "
              f"{len(drift)} changed" + (f", {len(missing)} now absent" if missing else ""))
        for k in drift[:8]:
            extra = ""
            if f == "coverage":
                extra = (f"   word_recall {b[k].get('word_recall')} -> {c[k].get('word_recall')}"
                         f"  numeral {b[k].get('numeral_recall')} -> {c[k].get('numeral_recall')}")
            print(f"           {k[:60]}{extra}")
        if drift:
            fail.append(f"{label}: {len(drift)} changed")
        if missing:
            fail.append(f"{label}: {len(missing)} document(s) stopped producing a value")

    errs = [k for k in shared if any(x in c[k] for x in ("error", "coverage_error", "xcheck_error"))]
    berrs = [k for k in shared if any(x in b[k] for x in ("error", "coverage_error", "xcheck_error"))]
    if len(errs) != len(berrs):
        fail.append(f"error count moved: {len(berrs)} -> {len(errs)}")
    print(f"  {'OK   ' if len(errs) == len(berrs) else 'DRIFT'}  {'documents raising':<28} "
          f"{len(berrs)} -> {len(errs)}")

    print()
    if fail:
        print(f"GATE A2 FAILED — {len(fail)} finding(s):")
        for f_ in fail:
            print(f"  - {f_}")
        return 1
    print("QUALITY SNAPSHOT MATCHES")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
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
    n_cov = sum(1 for d in current["documents"].values() if d.get("coverage"))
    n_err = sum(1 for d in current["documents"].values()
                if any(x in d for x in ("error", "coverage_error", "xcheck_error")))
    print(f"wrote {out}")
    print(f"  coverage computed : {n_cov}/{current['document_count']}")
    print(f"  documents raising : {n_err}")


if __name__ == "__main__":
    main()
