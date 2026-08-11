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
    """`expect` is what repair_signs must produce WITH MUTATION ENABLED.

    The pass is flag-only by default, so every case is run with `apply=True`: a guard that
    only holds because mutation is switched off is not a guard. The default-off behaviour
    is asserted separately at the end.
    """
    audit: list[dict] = []
    out, _ = repair.repair_signs(text, src, audit=audit, apply=True)
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

# --- MATHS IS NEVER TOUCHED --------------------------------------------------------
# Every case below was a real shipped corruption before 2026-08-11. R10 is a PROSE
# defect: inside a math span the minus is markup the VLM emits explicitly, so a bare
# digit there is an exponent, an index, a numerator or a range endpoint.
print("\nmaths spans are inviolable (all of these shipped corrupted):")

for name, text, src in [
    ("\\frac numerator", r"$J = - \frac { 1 } { k } e$", "J = −1/k and a value −1"),
    ("bracketed expression", r"$\mathbb { R } ^ { ( k + 1 ) d }$", "R^{(k+1)d}, delta −1"),
    ("subscript index", r"$N _ { 1 } = N _ { 2 }$", "N_1 = N_2, shift −1"),
    ("en-dash range in maths", r"Layers $2 { - } 4 $ here", "Layers 2-4, delta −4"),
    ("display maths", r"$$x = 1 + y$$", "x = −1 + y"),
]:
    check(name, text, src, expect=text)

check("percent range is not a dropped sign", "spreads of 3% 4%",
      "spreads 3% to 4%, change −4%", expect="spreads of 3% 4%")
check("a sign inside an HTML tag counts as already-signed", "<sub>−</sub>0.0028",
      "value −0.0028 in table", expect="<sub>−</sub>0.0028")

# --- FLAG-ONLY BY DEFAULT ----------------------------------------------------------
print("\nflag-only by default (value matching cannot prove the TARGET is signed):")

audit: list[dict] = []
out, n = repair.repair_signs("an R2 of 0.47%", "an R2 of −0.47%", audit=audit)
if out == "an R2 of 0.47%" and n == 0 and [a.get("action") for a in audit] == ["flagged"]:
    print("  ok   a candidate is FLAGGED, not applied, unless mutation is requested")
else:
    FAILURES.append("flag-only default")
    print(f"  FAIL flag-only default: out={out!r} n={n} audit={audit!r}")

audit = []
repair.repair_signs("an R2 of 0.47%", "an R2 of −0.47%", audit=audit, apply=True)
if [a.get("action") for a in audit] == ["signed"] and "context" in audit[0]:
    print("  ok   an applied edit is recorded with its context")
else:
    FAILURES.append("signed edits are recorded")
    print(f"  FAIL signed edits are recorded: {audit!r}")

audit = []
repair.repair_signs("layer shape [2,128]", "shift of −2,128 units", audit=audit, apply=True)
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
