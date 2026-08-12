#!/usr/bin/env python3
"""Guards snapshot.compare() — the Gate A verdict function.

WHY THIS EXISTS. Gate A licenses refactors: it rebuilds all 636 documents offline and
compares hashes. That makes compare() the single function standing between a refactor
and the corpus, and on 2026-08-12 it was audited and found to report success in four
situations where the corpus had genuinely changed:

  1. documents present only in the CURRENT run were printed but never counted;
  2. a field that was None on BOTH sides compared equal, so "no evidence" read as "agreed";
  3. the producer block (render DPI, cap, MinerU version) was printed but never compared —
     mutation testing measured this exactly: EXPECTED_DPI 300 -> 72 moved 0 of 200
     rebuild hashes, because an offline rebuild never rasterises a page;
  4. source_pdf_found was recorded and ignored, so losing PDF resolution — which silently
     disables repair and salvage — was invisible for any document needing no repairs.

Every case below is one of those, plus the benign cases that must NOT fail (an absolute
path moving, the rebuild code changing — which is the whole point of a refactor gate).

Run:  .venv/bin/python tests/local_deploy/test_snapshot_compare.py
"""
from __future__ import annotations

import copy
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "local_deploy"))

import snapshot  # noqa: E402

FAILURES: list[str] = []


def ok(name: str) -> None:
    print(f"  ok    {name}")


def fail(name: str, detail: str) -> None:
    FAILURES.append(name)
    print(f"  FAIL  {name}: {detail}")


def _snap() -> dict:
    """A minimal healthy snapshot: two documents, everything resolved."""
    return {
        "schema_version": snapshot.SCHEMA_VERSION,
        "producer": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "rebuild_code_sha256": "c0de",
            "rebuild_code_files": ["convert.py", "postprocess.py"],
            "render": {"dpi": 300, "page_to_image_defaults": [300, 4500],
                       "mineru_version": "3.4.4",
                       "mineru_path": "/Users/knight/code_local/minerU_pipeline/vendor/mineru/mineru"},
        },
        "document_count": 2,
        "documents": {
            "papers/alpha": {"track": "papers", "md_sha256": "m1", "content_list_sha256": "c1",
                             "qa_sha256": "q1", "rebuild_sha256": "r1", "source_pdf_found": True},
            "papers/beta": {"track": "papers", "md_sha256": "m2", "content_list_sha256": "c2",
                            "qa_sha256": "q2", "rebuild_sha256": "r2", "source_pdf_found": True},
        },
    }


def verdict(base: dict, cur: dict, **kw) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = snapshot.compare(base, cur, **kw)
    return rc, buf.getvalue()


def expect(name: str, base: dict, cur: dict, want: int, needle: str | None = None, **kw) -> None:
    rc, out = verdict(base, cur, **kw)
    if rc != want:
        fail(name, f"expected exit {want}, got {rc}\n{out}")
        return
    if needle and needle.lower() not in out.lower():
        fail(name, f"exit {rc} correct but the report never mentions {needle!r}\n{out}")
        return
    ok(name)


print("snapshot.compare() — the four blind spots found on 2026-08-12")

# --- 1. new documents were printed but not counted -----------------------------------
b = _snap(); c = copy.deepcopy(b)
c["documents"]["papers/gamma"] = {"track": "papers", "md_sha256": "m3",
                                  "content_list_sha256": "c3", "qa_sha256": "q3",
                                  "rebuild_sha256": "r3", "source_pdf_found": True}
expect("a document absent from the baseline fails the gate", b, c, 1, "new")
expect("...and is tolerated with --allow-new", b, c, 0, allow_new=True)

# --- 2. None == None read as agreement ------------------------------------------------
b = _snap(); c = copy.deepcopy(b)
b["documents"]["papers/alpha"]["rebuild_sha256"] = None
c["documents"]["papers/alpha"]["rebuild_sha256"] = None
expect("a document that rebuilds to nothing on BOTH sides is not 'OK'", b, c, 1, "unusable")

b = _snap(); c = copy.deepcopy(b)
b["documents"]["papers/alpha"]["md_sha256"] = None
c["documents"]["papers/alpha"]["md_sha256"] = None
expect("a .md missing on BOTH sides is not 'OK'", b, c, 1, "unusable")

# --- 3. the producer block was never compared ----------------------------------------
for key, bad in (("dpi", 72), ("page_to_image_defaults", [300, 3500]),
                 ("mineru_version", "4.0.0a5")):
    b = _snap(); c = copy.deepcopy(b)
    c["producer"]["render"][key] = bad
    expect(f"render.{key} regression is caught", b, c, 1, f"render.{key}")

# ...but an absolute path moving is benign and must NOT fail. It already did move once,
# on 2026-08-11, when the whole tree was relocated.
b = _snap(); c = copy.deepcopy(b)
c["producer"]["render"]["mineru_path"] = "/somewhere/else/mineru"
expect("a moved mineru_path does NOT fail the gate", b, c, 0)

# EVERY real comparison straddles the JSON boundary: the baseline is loaded from a file
# (tuple -> list) while the current side is live Python from fork_render_settings(). The
# first real run of the hardened gate failed on exactly this, with all 636 documents
# clean — a false positive that would have fired forever. Unit tests built both sides in
# Python and never crossed that boundary, so this case has to be explicit.
b = _snap()
b["producer"]["render"]["page_to_image_defaults"] = [300, 4500]      # as JSON stores it
c = copy.deepcopy(b)
c["producer"]["render"]["page_to_image_defaults"] = (300, 4500)      # as Python returns it
expect("a tuple/list difference across the JSON boundary is NOT a regression", b, c, 0)

# ...and the normalisation must not blunt the real check it exists to serve.
c = copy.deepcopy(b)
c["producer"]["render"]["page_to_image_defaults"] = (300, 3500)
expect("...but a genuinely different cap still fails", b, c, 1, "page_to_image_defaults")

# --- 4. source_pdf_found was recorded and ignored -------------------------------------
b = _snap(); c = copy.deepcopy(b)
c["documents"]["papers/alpha"]["source_pdf_found"] = False
expect("losing source-PDF resolution is caught", b, c, 1, "source pdf")

b = _snap(); c = copy.deepcopy(b)
b["documents"]["papers/alpha"]["source_pdf_found"] = False
expect("GAINING source-PDF resolution is a note, not a failure", b, c, 0, "newly resolve")

print("\nthe cases that must still behave")

# --- identical -------------------------------------------------------------------------
b = _snap()
expect("identical snapshots pass", b, copy.deepcopy(b), 0, "SNAPSHOT MATCHES")

# --- the drift the original DID catch, still caught ------------------------------------
b = _snap(); c = copy.deepcopy(b)
c["documents"]["papers/beta"]["rebuild_sha256"] = "DIFFERENT"
expect("a changed rebuild hash still fails", b, c, 1, "GATE A")

b = _snap(); c = copy.deepcopy(b)
del c["documents"]["papers/beta"]
expect("a vanished document still fails", b, c, 1, "gone")

# --- a refactor changes the code hash; that is the POINT, not a failure ----------------
b = _snap(); c = copy.deepcopy(b)
c["producer"]["rebuild_code_sha256"] = "d1ff"
c["producer"]["git_commit"] = "b" * 40
c["producer"]["rebuild_code_files"] = ["build/markdown.py", "quality/repair.py"]
expect("changed rebuild code passes when output is identical", b, c, 0, "CHANGED")

# ...and identical code must SAY so, because then a pass proves nothing about a refactor.
b = _snap()
_, out = verdict(b, copy.deepcopy(b))
if "byte-identical to the baseline" in out:
    ok("an unchanged-code run says the pass proves nothing about a refactor")
else:
    fail("an unchanged-code run says the pass proves nothing about a refactor", out)

# --- a dirty baseline is a warning, not a failure --------------------------------------
# It bit us on 2026-08-12: baseline-20260811 was captured from a dirty tree and so could
# not be reproduced from any commit. Worth saying loudly; not worth blocking on.
b = _snap(); b["producer"]["git_dirty"] = True
expect("a dirty baseline warns but does not block", b, copy.deepcopy(_snap()), 0, "DIRTY")

# --- schema mismatch is uncomparable, not merely different -----------------------------
b = _snap(); c = copy.deepcopy(b)
c["schema_version"] = "corpus-snapshot-v2"
expect("an incomparable schema exits 2, distinct from a drift exit of 1", b, c, 2)

# --- a new rebuild exception ------------------------------------------------------------
b = _snap(); c = copy.deepcopy(b)
c["documents"]["papers/alpha"]["rebuild_sha256"] = None
c["documents"]["papers/alpha"]["rebuild_error"] = "KeyError: 'text'"
expect("a newly-raising rebuild fails", b, c, 1, "rebuild")

# --- _producer() must never silently hash nothing ---------------------------------------
# After W3 these modules move. A silent empty hash would read as "the code never changed"
# at exactly the moment the code moved.
_real_here = snapshot._HERE
try:
    snapshot._HERE = Path("/nonexistent/there/are/no/modules/here")
    try:
        snapshot._producer()
        fail("_producer() refuses to hash an empty module set", "it returned instead of raising")
    except SystemExit as e:
        if "hash of nothing" in str(e):
            ok("_producer() refuses to hash an empty module set")
        else:
            fail("_producer() refuses to hash an empty module set", f"wrong message: {e}")
finally:
    snapshot._HERE = _real_here

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    raise SystemExit(1)
print("All snapshot.compare() goldens passed.")
