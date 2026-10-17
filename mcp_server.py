"""Local MCP (Model Context Protocol) server for the Groww review workflow.

Exposes exactly two tools over stdio:

    append_to_notes      -> appends the weekly pulse to Notes.md
    create_email_draft   -> writes an .eml draft file (never sends anything)

This server never sends email and never makes outbound network calls. It only
writes to the local ``Notes.md`` file and the local ``drafts/`` directory.
Both are created only when a tool is actually called -- i.e. only after the
human clicks "Approve & Execute" in the Streamlit app.

Run standalone for debugging (it will wait on stdio):

    python mcp_server.py
"""

from __future__ import annotations

import json
import re
from datetime import date as _date
from email.message import EmailMessage
from pathlib import Path

from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
NOTES_FILE = BASE_DIR / "Notes.md"
DRAFTS_DIR = BASE_DIR / "drafts"

MAX_FEE_BULLETS = 6

mcp = FastMCP("groww-review-actions")


def _ok(**kwargs) -> str:
    return json.dumps({"status": "ok", **kwargs}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _safe_date(value: str) -> str:
    """Filesystem-safe ISO date, defaulting to today."""
    if value:
        m = re.search(r"\d{4}-\d{2}-\d{2}", str(value))
        if m:
            return m.group(0)
    return _date.today().isoformat()


@mcp.tool()
def append_to_notes(
    date: str,
    top_themes: list[str],
    weekly_pulse: str,
    identified_fee_issue: str,
    explanation_bullets: list[str],
    source_links: list[str],
) -> str:
    """Append a weekly review digest to Notes.md.

    Args:
        date: The window/run date, e.g. "2026-10-15".
        top_themes: Short theme descriptions, most frequent first.
        weekly_pulse: A <=250 word narrative summary of the week.
        identified_fee_issue: One-line description of the dominant fee issue.
        explanation_bullets: <=6 short customer-facing explanation bullets.
        source_links: Official Groww source URLs backing the explanation.

    Returns:
        JSON status string describing what was written.
    """
    if not str(weekly_pulse).strip():
        return _err("weekly_pulse must not be empty")
    if len(explanation_bullets or []) > MAX_FEE_BULLETS:
        return _err(f"explanation_bullets must have at most {MAX_FEE_BULLETS} items")

    day = _safe_date(date)

    lines = [f"\n## Weekly Review Digest - {day}\n"]

    lines.append("### Top Themes")
    if top_themes:
        lines.extend(f"- {t}" for t in top_themes)
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("### Weekly Pulse")
    lines.append(weekly_pulse.strip())
    lines.append("")

    lines.append("### Identified Fee Issue")
    lines.append(str(identified_fee_issue).strip() or "(none)")
    lines.append("")

    lines.append("### Customer-facing Fee Explanation")
    if explanation_bullets:
        lines.extend(f"- {b}" for b in explanation_bullets)
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("### Sources")
    if source_links:
        lines.extend(f"- {u}" for u in source_links)
    else:
        lines.append("- (none)")
    lines.append("")

    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTES_FILE, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return _ok(message=f"Appended weekly digest for {day} to {NOTES_FILE.name}.",
               path=str(NOTES_FILE))


@mcp.tool()
def create_email_draft(subject: str, body: str) -> str:
    """Create an email draft on disk as an .eml file. Never sends.

    Args:
        subject: The email subject line.
        body: The plain-text email body.

    Returns:
        JSON status string with the path of the draft that was written.
    """
    if not str(subject).strip() or not str(body).strip():
        return _err("subject and body must not be empty")

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    msg = EmailMessage()
    msg["Subject"] = subject.strip()
    # Intentionally omit From/To/Date: nothing should be sendable directly and
    # no recipient is fabricated.
    msg.set_content(body)

    day = _date.today().isoformat()
    path = DRAFTS_DIR / f"{day}.eml"
    n = 1
    while path.exists():  # same-day reruns don't clobber earlier drafts
        n += 1
        path = DRAFTS_DIR / f"{day}-{n}.eml"

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(msg.as_string())

    return _ok(message=f"Email draft written to {path.name} (not sent).",
               path=str(path))


if __name__ == "__main__":
    mcp.run()  # stdio transport -- what the Streamlit client uses
