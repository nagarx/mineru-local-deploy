"""Regression guard for the non-BMP maths-variable fix (LOCAL FORK PATCH).

WHY THIS FILE EXISTS
--------------------
`mineru/utils/pdf_text_tool.py::_merge_surrogate_pairs` is a fork-local patch with no
upstream equivalent. Two ordinary, well-intentioned actions silently destroy it:

  1. `git merge upstream/master` — upstream rewrites this file (it gained 195 lines in
     3.4.4 alone), and the patch is easy to drop while resolving.
  2. `uv sync` / `pip install -U pdftext` — pdftext >=0.7 rewrites every UTF-16 surrogate
     to U+FFFD *inside* get_chars, so the halves never reach us and no downstream repair
     can recover them.

Either one reintroduces corrupted mathematics into every extracted paper, silently and
without any error. The '??' seen corpus-wide before the fix was 1,187 occurrences.
This test makes both failures loud.

Run standalone (no pytest needed):
    .venv/bin/python tests/local_deploy/test_surrogate_pairs.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mineru.utils.pdf_text_tool import _merge_surrogate_pairs  # noqa: E402

# U+1D44B MATHEMATICAL ITALIC CAPITAL X == surrogate pair D835 DC4B
HI, LO, EXPECTED = 0xD835, 0xDC4B, 0x1D44B


def _c(codepoint, idx=0):
    """A minimal pdftext-shaped char dict."""
    return {"char": chr(codepoint), "bbox": [0.0, 0.0, 1.0, 1.0],
            "rotation": 0.0, "font": {"name": "F"}, "char_idx": idx}


def test_valid_pair_is_decoded():
    """The whole point: two surrogate halves are ONE character, not two unknowns."""
    out = _merge_surrogate_pairs([_c(HI, 0), _c(LO, 1)])
    assert len(out) == 1, f"pair must collapse to one char, got {len(out)}"
    assert ord(out[0]["char"]) == EXPECTED, (
        f"expected U+{EXPECTED:04X} ({chr(EXPECTED)}), got U+{ord(out[0]['char']):04X}")
    assert out[0]["bbox"] == [0.0, 0.0, 1.0, 1.0], "first half's bbox must be kept verbatim"


def test_pair_among_ordinary_text():
    """Ordinary characters must pass through untouched and stay in order."""
    chars = [_c(ord("f"), 0), _c(HI, 1), _c(LO, 2), _c(ord("="), 3)]
    out = "".join(c["char"] for c in _merge_surrogate_pairs(chars))
    assert out == f"f{chr(EXPECTED)}=", repr(out)


def test_unpairable_half_becomes_replacement_char():
    """A broken ToUnicode CMap yields a half with no partner (real case: ChronosX p3,
    U+D835 followed by '!'). It is unrecoverable AND not UTF-8 encodable, so it must be
    neutralised here rather than crashing a writer or vanishing silently."""
    for chars in ([_c(HI, 0), _c(ord("!"), 1)],   # high with no low
                  [_c(LO, 0), _c(ord("!"), 1)],   # stray low
                  [_c(HI, 0)]):                   # high at end of page
        out = _merge_surrogate_pairs(chars)
        text = "".join(c["char"] for c in out)
        assert not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text), (
            f"a lone surrogate escaped: {text!r}")
        assert text.startswith("�"), f"expected U+FFFD marker, got {text!r}"
        text.encode("utf-8")  # must not raise


def test_shape_defensive():
    """If pdftext's return shape changes, degrade to a no-op instead of raising.
    Blocking a bad pdftext is preflight_deps()'s job, not this function's."""
    for weird in ([], None, "PageChars-like", [object()]):
        assert _merge_surrogate_pairs(weird) is weird


def test_pdftext_still_hands_us_the_surrogate_halves():
    """VERSION-AGNOSTIC dependency guard.

    Asserts the behaviour we depend on rather than a version string: pdftext must not
    pre-replace surrogates. pdftext >=0.7 does exactly that (`if 0xD800 <= code <= 0xDFFF:
    code = 0xFFFD`), which destroys every non-BMP maths glyph before we can decode it.
    """
    import inspect

    from pdftext.pdf import chars as pdftext_chars

    src = inspect.getsource(pdftext_chars.get_chars)
    assert "0xFFFD" not in src.upper().replace("0XFFFD", "0xFFFD"), (
        "This pdftext replaces surrogates with U+FFFD inside get_chars, destroying every "
        "non-BMP mathematical variable before MinerU can decode it. Pin pdftext<0.7 "
        "(see the block comment in pyproject.toml).")


def test_real_pdf_recovers_maths_glyphs():
    """End-to-end on a real maths-heavy paper, if the local library is present."""
    import glob
    # The corpus moved out of the fork on 2026-08-11. MINERU_PIPELINE_DATA is the one
    # resolution point; this test degrades to "skipped" rather than failing without it.
    import os
    data = Path(os.environ.get("MINERU_PIPELINE_DATA",
                               Path.home() / "code_local" / "scriptorium"))
    hits = glob.glob(str(data / "corpora/*/inbox/*.pdf")) + \
        glob.glob(str(data / "corpora/*/done/*.pdf"))
    if not hits:
        print("  ~ skipped (no local library PDFs available)")
        return

    import pypdfium2 as pdfium

    from mineru.utils.pdf_text_tool import get_page_chars
    for pdf in sorted(hits):
        doc = pdfium.PdfDocument(pdf)
        try:
            for pi in range(min(len(doc), 8)):
                text = "".join(c.get("char", "") for c in get_page_chars(doc[pi])["chars"])
                assert not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text), (
                    f"lone surrogate escaped in {Path(pdf).name} p{pi}")
                text.encode("utf-8")  # must never raise
                if any(ord(ch) > 0xFFFF for ch in text):
                    print(f"  ~ recovered non-BMP maths in {Path(pdf).name[:44]} p{pi}")
                    return
        finally:
            doc.close()
    print("  ~ no non-BMP glyphs in the sampled pages (nothing to prove, nothing broken)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}\n      {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
