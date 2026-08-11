#!/usr/bin/env python3
"""Guards the ligature fork patch in mineru/utils/pdf_text_tool.py.

Upstream's `_deduplicate_near_identical_chars` removes a visible character whose same
character was already seen at a near-identical bbox. A ligature is ONE glyph whose
ToUnicode expands to SEVERAL characters, all carrying that glyph's bbox — so the second
'f' of "different" was deleted, emitting "diferent". Measured over 636 documents on
2026-08-11: 267 files (42%), 8,618 occurrences, 267/551 native-text vs 0/85 VLM-OCR.
Not f-only: 'tt' collapsed in 164 further files ("attention" -> "atention").

`_is_same_glyph_expansion` refuses exactly that deletion. This file fails loudly if the
patch is lost in an upstream merge, or if its discriminator stops working.

Run:  .venv/bin/python tests/local_deploy/test_ligature_dedup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "local_deploy"))

import mineru.utils.pdf_text_tool as ptt  # noqa: E402

FAILURES: list[str] = []


def ok(name: str) -> None:
    print(f"  ok   {name}")


def fail(name: str, detail: str) -> None:
    FAILURES.append(name)
    print(f"  FAIL {name}\n       {detail}")


def _char(ch: str, bbox: tuple[float, float, float, float]) -> dict:
    """A char dict shaped like pdftext's, with the keys the deduper reads."""
    return {"char": ch, "bbox": list(bbox), "rotation": 0,
            "font": {"name": "TestFont", "size": 9.0, "weight": 400}}


print("the patch must be present at all:")
if hasattr(ptt, "_is_same_glyph_expansion"):
    ok("_is_same_glyph_expansion exists")
else:
    fail("_is_same_glyph_expansion exists",
         "FORK PATCH LOST — every ligature will silently lose a character")
    print("\n1 FAILURE(S)")
    raise SystemExit(1)

print("\nligature constituents must SURVIVE (the corruption this patch fixes):")

# 'ff' of "different": one glyph, so both chars carry the SAME bbox.
BB = (100.0, 200.0, 105.0, 210.0)
chars = [_char("d", (90.0, 200.0, 95.0, 210.0)), _char("f", BB), _char("f", BB)]
out = "".join(c["char"] for c in ptt._deduplicate_near_identical_chars(chars))
if out == "dff":
    ok("'ff' ligature keeps both characters")
else:
    fail("'ff' ligature keeps both characters", f"expected 'dff', got {out!r}")

# 'tt' of "attention" — measured bbox delta was exactly [0,0,0,0], same as 'ff'.
chars = [_char("a", (90.0, 200.0, 95.0, 210.0)), _char("t", BB), _char("t", BB)]
out = "".join(c["char"] for c in ptt._deduplicate_near_identical_chars(chars))
if out == "att":
    ok("'tt' ligature keeps both characters")
else:
    fail("'tt' ligature keeps both characters", f"expected 'att', got {out!r}")

print("\ngenuine duplicates must STILL be removed (the patch must stay narrow):")

# A text-layer boundary duplicate re-emits a character seen EARLIER and elsewhere,
# so it is not adjacent in the kept stream. That is what this function exists for.
chars = [_char("x", BB), _char("y", (200.0, 200.0, 205.0, 210.0)), _char("x", BB)]
out = "".join(c["char"] for c in ptt._deduplicate_near_identical_chars(chars))
if out == "xy":
    ok("non-adjacent duplicate is still removed")
else:
    fail("non-adjacent duplicate is still removed", f"expected 'xy', got {out!r}")

# A different character at the same box is not a duplicate at all.
chars = [_char("f", BB), _char("i", BB)]
out = "".join(c["char"] for c in ptt._deduplicate_near_identical_chars(chars))
if out == "fi":
    ok("distinct characters at one bbox are both kept")
else:
    fail("distinct characters at one bbox are both kept", f"expected 'fi', got {out!r}")

# NEAR-identical but NOT exactly equal must still dedupe: only a bit-for-bit equal
# bbox is the ligature signature (measured delta [0.0, 0.0, 0.0, 0.0]).
near = (100.4, 200.4, 105.4, 210.4)          # inside NEAR_IDENTICAL tolerance of 1.0
chars = [_char("f", BB), _char("f", near)]
out = "".join(c["char"] for c in ptt._deduplicate_near_identical_chars(chars))
if out == "f":
    ok("adjacent but merely NEAR-identical bbox is still deduped")
else:
    fail("adjacent but merely NEAR-identical bbox is still deduped",
         f"expected 'f', got {out!r} — the discriminator is too loose")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    raise SystemExit(1)
print("All ligature-dedup goldens passed.")
