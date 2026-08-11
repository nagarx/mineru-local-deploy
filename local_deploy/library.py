"""
library.py — a content-hash ledger for cyclic, resumable, never-miss PDF processing.

Every PDF that ever enters the inbox is tracked by the SHA-256 of its BYTES, so renames,
moves, and duplicates never cause a re-process or a miss. The ledger (SQLite, WAL mode,
stdlib) is the single source of truth: at any time  done + pending + in_progress + failed
== total registered. Nothing silently vanishes.

Status lifecycle:  pending -> in_progress -> done | failed(+attempts) ; failed retried up
to MAX_ATTEMPTS, then it stays failed and is surfaced in report.md (never dropped).
A crash between next_batch() and mark_*() leaves rows 'in_progress'; reset_stuck() returns
them to 'pending' on the next run (idempotent — outputs are overwritten atomically).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PDF_SUFFIXES = {".pdf"}
MAX_ATTEMPTS = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.")
    return (s or "doc")[:120]


class Library:
    #: `output_md` is stored RELATIVE to the track root and resolved at read time, so the
    #: whole data root can be relocated without a migration. Absolute paths were the
    #: original design and they broke twice: `_archive_first_corpus` sat 137/137 unverifiable
    #: for a month, and the 2026-08-11 relocation invalidated all 636 rows at once. The
    #: value is fully derivable (`output/<slug>.md`), so storing it absolutely bought
    #: nothing and cost relocatability.
    SCHEMA_VERSION = 2

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.root = self.db_path.parent.parent      # <track>/state/ledger.db -> <track>
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS papers(
                id            TEXT PRIMARY KEY,      -- sha256(bytes)[:16]
                source_path   TEXT,                 -- latest inbox path seen
                slug          TEXT UNIQUE,           -- output basename (readable, unique)
                status        TEXT NOT NULL DEFAULT 'pending',
                needs_review  INTEGER DEFAULT 0,
                attempts      INTEGER DEFAULT 0,
                error         TEXT,
                pages         INTEGER,
                output_md     TEXT,
                registered_at TEXT,
                updated_at    TEXT)"""
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- registration ---------------------------------------------------------

    def _unique_slug(self, base: str, fid: str) -> str:
        # try increasingly-specific candidates; the full 16-hex id is the PK so it's
        # globally unique. Loop guarantees we never emit a taken slug (no IntegrityError).
        for cand in (base, f"{base}-{fid[:8]}", f"{base}-{fid}"):
            if not self.conn.execute("SELECT 1 FROM papers WHERE slug=?", (cand,)).fetchone():
                return cand
        i = 2
        while self.conn.execute("SELECT 1 FROM papers WHERE slug=?", (f"{base}-{fid}-{i}",)).fetchone():
            i += 1
        return f"{base}-{fid}-{i}"

    def register_inbox(self, inbox_dir: Path | str) -> int:
        """Hash every PDF under inbox (recursive) and register NEW content as pending.
        Returns count of newly-registered PDFs. Idempotent. A file that can't be read
        (mid-copy, locked, raced away) or a rare clash is SKIPPED this scan — never aborts
        the walk — so it is simply re-picked on the next scan once it's readable."""
        inbox = Path(inbox_dir)
        if not inbox.exists():
            return 0
        new, skipped = 0, 0
        for p in sorted(inbox.rglob("*")):
            if not (p.is_file() and p.suffix.lower() in PDF_SUFFIXES):
                continue
            try:
                fid = hash_file(p)
                if self.conn.execute("SELECT id FROM papers WHERE id=?", (fid,)).fetchone():
                    self.conn.execute("UPDATE papers SET source_path=?, updated_at=? WHERE id=?",
                                      (str(p), _now(), fid))
                    continue
                slug = self._unique_slug(slugify(p.stem), fid)
                self.conn.execute(
                    "INSERT INTO papers(id,source_path,slug,status,registered_at,updated_at) "
                    "VALUES(?,?,?,'pending',?,?)", (fid, str(p), slug, _now(), _now()))
                new += 1
            except (OSError, sqlite3.IntegrityError) as e:
                skipped += 1
                print(f"  [register] skipped '{p.name}' ({type(e).__name__}); will retry next scan",
                      flush=True)
        self.conn.commit()
        if skipped:
            print(f"  [register] {skipped} file(s) skipped this scan (re-tried automatically next scan)",
                  flush=True)
        return new

    # --- queue ----------------------------------------------------------------

    def reset_stuck(self) -> int:
        cur = self.conn.execute(
            "UPDATE papers SET status='pending', updated_at=? WHERE status='in_progress'", (_now(),))
        self.conn.commit()
        return cur.rowcount

    def requeue_failed(self) -> int:
        cur = self.conn.execute(
            "UPDATE papers SET status='pending', attempts=0, updated_at=? WHERE status='failed'", (_now(),))
        self.conn.commit()
        return cur.rowcount

    def next_batch(self, n: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id,source_path,slug FROM papers "
            "WHERE status='pending' OR (status='failed' AND attempts < ?) "
            "ORDER BY registered_at LIMIT ?", (MAX_ATTEMPTS, n)).fetchall()
        for r in rows:
            self.conn.execute("UPDATE papers SET status='in_progress', updated_at=? WHERE id=?",
                              (_now(), r[0]))
        self.conn.commit()
        return [{"id": r[0], "source_path": r[1], "slug": r[2]} for r in rows]

    # --- results --------------------------------------------------------------

    def _relativise(self, path: str | Path) -> str:
        """Store paths under the track root as relative; leave anything else alone."""
        p = Path(path)
        if not p.is_absolute():
            return str(p)
        try:
            return str(p.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(p)

    def resolve_output(self, output_md: str | None) -> Path | None:
        """Read-side counterpart of _relativise."""
        if not output_md:
            return None
        p = Path(output_md)
        return p if p.is_absolute() else self.root / p

    def mark_done(self, fid: str, output_md: str, needs_review: bool, pages: int | None = None) -> None:
        self.conn.execute(
            "UPDATE papers SET status='done', needs_review=?, output_md=?, pages=?, "
            "error=NULL, updated_at=? WHERE id=?",
            (1 if needs_review else 0, self._relativise(output_md), pages, _now(), fid))
        self.conn.commit()

    def mark_failed(self, fid: str, error: str) -> None:
        self.conn.execute(
            "UPDATE papers SET status='failed', attempts=attempts+1, error=?, updated_at=? WHERE id=?",
            (str(error)[:1000], _now(), fid))
        self.conn.commit()

    # --- reporting / verification --------------------------------------------

    def counts(self) -> dict:
        c = {s: 0 for s in ("pending", "in_progress", "done", "failed")}
        for s, n in self.conn.execute("SELECT status, COUNT(*) FROM papers GROUP BY status"):
            c[s] = n
        c["total"] = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]  # authoritative
        c["needs_review"] = self.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE status='done' AND needs_review=1").fetchone()[0]
        return c

    def verify_outputs(self) -> list[str]:
        """Every 'done' paper must have its output_md on disk. Returns slugs of any that don't."""
        missing = []
        for slug, omd in self.conn.execute(
                "SELECT slug, output_md FROM papers WHERE status='done'"):
            p = self.resolve_output(omd)
            if p is None or not p.exists():
                missing.append(slug)
        return missing

    def migrate_paths(self, dry_run: bool = False) -> dict[str, int]:
        """Make stored paths relocatable, and re-home any that point at a vanished tree.

        Idempotent. `output_md` becomes `output/<slug>.md` — the invariant the writer has
        always used — so it survives any future move of the data root. `source_path` is
        re-pointed into this track when its recorded location no longer exists; it is
        transient anyway (register_inbox rewrites it every scan), but leaving it dangling
        makes a requeue fail instantly.
        """
        stats = {"output_md_relativised": 0, "source_path_rehomed": 0, "source_path_missing": 0}
        rows = self.conn.execute("SELECT id, slug, source_path, output_md FROM papers").fetchall()
        # Index this track first, then sibling tracks — the same order find_source_pdf uses.
        # Siblings matter: 103 of research_papers' 302 source PDFs physically live under
        # _archive_first_corpus, so a track-only scan leaves a third of the corpus dangling.
        by_name: dict[str, Path] = {}

        def _index(base: Path) -> None:
            for sub in ("inbox", "done", "_originals"):
                d = base / sub
                if d.is_dir():
                    for p in d.rglob("*.pdf"):
                        by_name.setdefault(p.name, p)

        _index(self.root)
        if self.root.parent.is_dir():
            for sibling in sorted(self.root.parent.iterdir()):
                if sibling.is_dir() and sibling != self.root:
                    _index(sibling)
        for fid, slug, src, omd in rows:
            want_omd = f"output/{slug}.md"
            if omd != want_omd:
                stats["output_md_relativised"] += 1
                if not dry_run:
                    self.conn.execute("UPDATE papers SET output_md=? WHERE id=?", (want_omd, fid))
            if src and not Path(src).exists():
                hit = by_name.get(Path(src).name)
                if hit is not None:
                    stats["source_path_rehomed"] += 1
                    if not dry_run:
                        self.conn.execute("UPDATE papers SET source_path=? WHERE id=?",
                                          (str(hit), fid))
                else:
                    stats["source_path_missing"] += 1
        if not dry_run:
            self.conn.commit()
        return stats

    def write_report(self, report_path: Path | str) -> None:
        c = self.counts()
        failed = self.conn.execute(
            "SELECT slug,source_path,error,attempts FROM papers WHERE status='failed' ORDER BY slug").fetchall()
        review = self.conn.execute(
            "SELECT slug,output_md FROM papers WHERE status='done' AND needs_review=1 ORDER BY slug").fetchall()
        lines = [f"# Library status — {_now()}", "",
                 f"- **total registered:** {c['total']}",
                 f"- **done:** {c['done']}  (flagged for review: {c['needs_review']})",
                 f"- **pending:** {c['pending']}",
                 f"- **in progress:** {c['in_progress']}",
                 f"- **failed:** {c['failed']}", ""]
        if failed:
            root = Path(report_path).expanduser().resolve().parent
            lines.append(f"## ⚠ FAILED — inspect or `run.py --root {root} --retry-failed`")
            for slug, src, err, att in failed:
                lines.append(f"- **{slug}** (attempt {att}/{MAX_ATTEMPTS}): {(err or '')[:180]}")
                lines.append(f"    - source: `{src}`")
            lines.append("")
        if review:
            lines.append("## Done — flagged for human review "
                         "(see the MINERU-QA header + inline flags in each .md; full record in output/.sidecar/)")
            for slug, omd in review:
                lines.append(f"- `{slug}` → {omd}")
            lines.append("")
        Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- self-test (no real PDFs / models needed) ------------------------------------

if __name__ == "__main__":
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inbox = d / "inbox"; inbox.mkdir()
        # two distinct "PDFs" + one duplicate (same bytes, different name) + a subfolder one
        (inbox / "paper one.pdf").write_bytes(b"CONTENT-A")
        (inbox / "paper_two.pdf").write_bytes(b"CONTENT-B")
        (inbox / "dup of A.pdf").write_bytes(b"CONTENT-A")          # duplicate content
        (inbox / "sub").mkdir(); (inbox / "sub" / "paper one.pdf").write_bytes(b"CONTENT-C")  # same name, diff content

        lib = Library(d / "state" / "ledger.db")
        n1 = lib.register_inbox(inbox)
        assert n1 == 3, f"expected 3 distinct (A,B,C); dup collapsed -> got {n1}"
        assert lib.register_inbox(inbox) == 0, "re-scan must register 0 new (idempotent)"
        c = lib.counts(); assert c["total"] == 3 and c["pending"] == 3, c

        # slugs unique even for the same-name different-content pair
        slugs = [r[0] for r in lib.conn.execute("SELECT slug FROM papers").fetchall()]
        assert len(set(slugs)) == 3, f"slugs must be unique: {slugs}"

        b1 = lib.next_batch(2); assert len(b1) == 2, b1
        assert lib.counts()["in_progress"] == 2
        # crash simulation: reset returns them to pending
        assert lib.reset_stuck() == 2
        assert lib.counts()["pending"] == 3

        b = lib.next_batch(10); assert len(b) == 3
        lib.mark_done(b[0]["id"], str(d / "output" / (b[0]["slug"] + ".md")), needs_review=False, pages=4)
        lib.mark_done(b[1]["id"], str(d / "output" / (b[1]["slug"] + ".md")), needs_review=True, pages=7)
        lib.mark_failed(b[2]["id"], "boom")
        c = lib.counts()
        assert c == {"pending": 0, "in_progress": 0, "done": 2, "failed": 1, "total": 3, "needs_review": 1}, c

        # failed is retryable (attempts 1 < 3) -> reappears in next_batch
        assert len(lib.next_batch(10)) == 1, "failed(<MAX) should be retryable"
        # verify catches the missing output files (we never actually wrote them)
        assert set(lib.verify_outputs()) == {b[0]["slug"], b[1]["slug"]}

        lib.write_report(d / "report.md")
        assert (d / "report.md").exists()
        print("counts:", c)
        print("All library self-tests passed.")
