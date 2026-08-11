#!/usr/bin/env python3
"""Negative-golden suite for repair.py's sign pass — the only code that MUTATES content.

Every case below is a real defect this pipeline shipped, or a real repair it must keep
making. The negative cases matter more than the positive ones: signing a number that was
never negative inverts a reported result, happens BEFORE the sidecar is written (so the
original is unrecoverable), and until 2026-08-11 shipped under `verdict: clean`.

Run:  .venv/bin/python tests/local_deploy/test_repair_signs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "local_deploy"))

import repair  # noqa: E402

FAILURES: list[str] = []


def check(name: str, text: str, src: str, *, expect: str) -> None:
    """`expect` is the exact string repair_signs must produce."""
    audit: list[dict] = []
    out, _ = repair.repair_signs(text, src, audit=audit)
    if out == expect:
        print(f"  ok   {name}")
        return
    FAILURES.append(name)
    print(f"  FAIL {name}\n       expected: {expect!r}\n       actual  : {out!r}")


# --- MUST NOT SIGN -----------------------------------------------------------------
print("negative goldens (a signed number here is silent data corruption):")

# C2, measured 2026-08-11. Mamba's source reads Uniform([0.001, 0.1]); the shipped
# markdown read [ 0 . 0 0 −1 , 0 . −1 ] — BOTH literals destroyed, verdict: clean.
# MinerU emits inline maths space-separated, so the boundary guard (?<![\d.])1(?![\d])
# saw a SPACE before the final '1' of '0 . 0 0 1' and signed a digit mid-literal.
_MAMBA = (r"initialized to $\tau _ { \Delta } ^ { - 1 } "
          r"( \mathsf { U n i f o r m } ( [ 0 . 0 0 1 , 0 . 1 ] ) )$")
check("C2 spaced numeral '0 . 0 0 1' is not a standalone 1",
      _MAMBA, "initialized to τ−1 (Uniform([0.001, 0.1])) following", expect=_MAMBA)

check("C2 spaced numeral '0 . 1' is not a standalone 1",
      "value $0 . 1$ here", "value −1 elsewhere and 0.1 here", expect="value $0 . 1$ here")

# The 2026-07 reverted regression: a value can be negative SOMEWHERE while this
# occurrence is a cross-reference ordinal.
check("cross-reference 'Fig. 23' is an ordinal",
      "see Fig. 23 for detail", "shift of −23 units", expect="see Fig. 23 for detail")
check("cross-reference 'Table 4' is an ordinal",
      "as Table 4 shows", "a delta of −4 units", expect="as Table 4 shows")
check("cross-reference 'Section 5' is an ordinal",
      "in Section 5 we", "a move of −5 bps", expect="in Section 5 we")

# Bracketed comma-lists are shapes/indices/citations far more often than quantities.
# Declined rather than decided — recorded in the audit for review.
check("tensor shape '[2,128]' is not a signed quantity",
      "layer shape [2,128]", "shift of −2,128 units", expect="layer shape [2,128]")
check("citation '[12]' is not a signed quantity",
      "as shown in [12]", "a drop of −12 bps", expect="as shown in [12]")

# --- MUST STILL SIGN ---------------------------------------------------------------
print("\npositive goldens (the repair must keep working):")

check("genuine dropped minus in prose is restored",
      "an out-of-sample R2 of 0.47%", "an out-of-sample R2 of −0.47%",
      expect="an out-of-sample R2 of −0.47%")
check("a trailing sentence dot does not block the repair",
      "the value was 0.47. Next", "the value was −0.47. Next",
      expect="the value was −0.47. Next")
check("an already-signed number is not double-signed",
      "a value of −0.47 here", "a value of −0.47 here",
      expect="a value of −0.47 here")

# --- AUDITABILITY ------------------------------------------------------------------
print("\nauditability (an unrecorded edit is unrecoverable):")

audit: list[dict] = []
repair.repair_signs("an R2 of 0.47%", "an R2 of −0.47%", audit=audit)
if [a.get("action") for a in audit] == ["signed"] and "context" in audit[0]:
    print("  ok   every applied edit is recorded with its context")
else:
    FAILURES.append("signed edits are recorded")
    print(f"  FAIL signed edits are recorded: {audit!r}")

audit = []
repair.repair_signs("layer shape [2,128]", "shift of −2,128 units", audit=audit)
if [a.get("action") for a in audit] == ["declined"]:
    print("  ok   declined candidates are surfaced, not hidden")
else:
    FAILURES.append("declines are recorded")
    print(f"  FAIL declines are recorded: {audit!r}")

# --- summary -----------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    raise SystemExit(1)
print("All repair_signs goldens passed.")
