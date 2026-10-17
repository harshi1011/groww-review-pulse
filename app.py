"""Streamlit app: Groww weekly review pulse + customer fee explainer.

Pipeline
--------
Step 0 (pandas, no LLM)
    - auto-detect review / rating / date columns (UI can override),
    - keep the last N weeks relative to the NEWEST review in the file,
    - clean (drop empty / <15 chars / duplicates),
    - keyword-frequency evidence table,
    - bounded, truncated, stratified sample + token estimate.
Step 1 (LLM, JSON)  -> themes (max 5), top 3, 3 verbatim quotes, fee issue
Step 2 (LLM)        -> weekly pulse (<=250 words) from themes+quotes+fee only
Step 3 (LLM)        -> fee explainer (<=6 bullets, allow-listed URLs only)
Validation gates    -> approval is disabled unless every gate passes.
Approval gate       -> only "Approve & Execute" calls the MCP server, which
                       appends Notes.md and writes an .eml draft. Nothing is
                       written to disk before that click.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import random
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import config

from groq import AuthenticationError, Groq, RateLimitError

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------------------------------------------------------------------------
# API key / Groq client
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """Read the API key from Streamlit secrets, then the environment."""
    for source in ("API_KEY", "GROQ_API_KEY"):
        try:
            if source in st.secrets:
                return str(st.secrets[source]).strip()
        except Exception:
            pass
        value = os.environ.get(source)
        if value:
            return value.strip()
    return None


def get_groq_client() -> Groq | None:
    key = get_api_key()
    return Groq(api_key=key) if key else None


# ---------------------------------------------------------------------------
# Data prep (cached -- never cache LLM output)
# ---------------------------------------------------------------------------
def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    """First candidate that exists in the file, matched case-insensitively."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


@st.cache_data(show_spinner=False)
def load_raw(data: bytes | None) -> pd.DataFrame:
    """Read the uploaded CSV bytes, or fall back to the bundled file."""
    if data is not None:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    return pd.read_csv(config.REVIEWS_CSV, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame, text_col: str, rating_col: str, date_col: str | None) -> pd.DataFrame:
    """Clean the raw frame into text / rating / reviewed_at."""
    cols = [text_col, rating_col] + ([date_col] if date_col else [])
    out = df[cols].copy()
    out = out.rename(columns={text_col: "text", rating_col: "rating",
                              **({date_col: "reviewed_at"} if date_col else {})})

    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() >= config.MIN_REVIEW_CHARS]
    out = out.drop_duplicates(subset=["text"])

    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out.dropna(subset=["rating"])
    out["rating"] = out["rating"].astype(int).clip(1, 5)

    if date_col:
        out["reviewed_at"] = pd.to_datetime(out["reviewed_at"], errors="coerce", utc=True)
        out = out.dropna(subset=["reviewed_at"])

    return out.reset_index(drop=True)


def window_data(df: pd.DataFrame, weeks: int) -> tuple[pd.DataFrame, pd.Timestamp | None, int]:
    """Return (windowed reviews, newest date, weeks actually used).

    The window is relative to the newest review in the file, not wall-clock
    today. If fewer than MIN_REVIEWS_IN_WINDOW reviews fall inside, widen by
    one week at a time up to WINDOW_WEEKS_HARD_MAX.
    """
    if "reviewed_at" not in df.columns or df.empty:
        return df, None, weeks

    newest = df["reviewed_at"].max()
    used = weeks
    win = df[df["reviewed_at"] >= newest - timedelta(weeks=used)]
    while len(win) < config.MIN_REVIEWS_IN_WINDOW and used < config.WINDOW_WEEKS_HARD_MAX:
        used += 1
        win = df[df["reviewed_at"] >= newest - timedelta(weeks=used)]
    return win.sort_values("reviewed_at", ascending=False), newest, used


def keyword_table(win: pd.DataFrame) -> pd.DataFrame:
    """Keyword frequency over the full filtered window (Step 0 evidence)."""
    texts = win["text"].str.lower()
    rows = [
        {"keyword": kw, "reviews mentioning": int(texts.str.contains(re.escape(kw), regex=True).sum())}
        for kw in config.MONEY_KEYWORDS
    ]
    return pd.DataFrame(rows)


def build_sample(win: pd.DataFrame) -> tuple[str, list[str], int, int, int]:
    """Bounded sample for LLM call 1.

    Returns (prompt_text, chosen_texts, n_sent, n_total, token_estimate).
    ``chosen_texts`` are the exact truncated strings sent to the model, so the
    quote-verification corpus can be built from them unchanged. If the window
    has more than MAX_LLM_REVIEWS reviews, take a seeded stratified sample that
    always keeps 1- and 2-star reviews mentioning money keywords.
    """
    texts = win["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, config.REVIEW_TRUNC_CHARS)
    ratings = win["rating"]
    total = len(win)
    idx = list(win.index)

    if total > config.MAX_LLM_REVIEWS:
        money_re = re.compile("|".join(re.escape(k) for k in config.MONEY_KEYWORDS))
        must = [i for i in idx if ratings[i] <= 2 and money_re.search(str(texts[i]).lower())]
        must = must[: config.MAX_LLM_REVIEWS // 2]
        rest = [i for i in idx if i not in set(must)]
        rng = random.Random(42)  # seeded -> reproducible
        need = config.MAX_LLM_REVIEWS - len(must)
        chosen = must + rng.sample(rest, min(need, len(rest)))
    else:
        chosen = idx

    chosen = list(chosen)
    chosen_texts = [str(texts[i]) for i in chosen]
    lines = [f"[{ratings[i]}\u2605] {texts[i]}" for i in chosen]
    body = "\n".join(lines)
    tokens = len(body) // 4  # rough chars/4 estimate
    return body, chosen_texts, len(chosen), total, tokens


# ---------------------------------------------------------------------------
# LLM calls (Groq SDK, JSON only)
# ---------------------------------------------------------------------------
def call_groq_json(client: Groq, model: str, prompt: str) -> dict:
    """One JSON-only completion, with 429 backoff and a single parse retry."""
    last_err: Exception | None = None
    text = ""
    for attempt in range(4):  # 1 try + 3 retries on 429
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=config.TEMPERATURE,
                max_completion_tokens=config.MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": prompt + "\nReturn valid JSON only. No prose, no code fences.",
                }],
            )
            text = resp.choices[0].message.content or ""
            break
        except RateLimitError as exc:
            last_err = exc
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
        except AuthenticationError:
            raise RuntimeError("API key missing or invalid. Check .streamlit/secrets.toml.")
    else:
        raise RuntimeError(f"Groq rate limit persisted after retries ({last_err}). Try again shortly.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # one parse retry, explicitly asking for JSON
        resp = client.chat.completions.create(
            model=model,
            temperature=config.TEMPERATURE,
            max_completion_tokens=config.MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=[{"role": "user",
                       "content": prompt + "\nYour previous reply was not valid JSON. Return valid JSON only."}],
        )
        return json.loads(resp.choices[0].message.content or "{}")


_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
}


def _norm(s: str) -> str:
    """Lowercase, collapse whitespace and fold common unicode punctuation.

    Folding curly quotes/apostrophes/dashes means a model that "copies" a
    review with different typography still matches the source in code.
    """
    out = str(s)
    for a, b in _PUNCT_MAP.items():
        out = out.replace(a, b)
    return re.sub(r"\s+", " ", out).strip().lower()


def verify_quote(quote: str, corpus: set[str]) -> bool:
    """True iff the quote (normalised) is a substring of some source review."""
    q = _norm(quote).strip("\"'")
    if len(q) < 8:
        return False
    return any(q in text for text in corpus)


def build_quote_candidates(sample_texts: list[str], max_candidates: int = 60) -> list[str]:
    """Exact sentences (code-extracted) the model must choose quotes from.

    Because each candidate is a literal slice of a review we sent, any quote the
    model selects is verbatim by construction -- the model only picks indices.
    """
    sentences: list[str] = []
    for text in sample_texts:
        for s in re.split(r"(?<=[.!?])\s+", text):
            s = s.strip(" -\t")
            if 20 <= len(s) <= config.REVIEW_TRUNC_CHARS:
                sentences.append(s)
    seen: set[str] = set()
    uniq: list[str] = []
    for s in sentences:
        key = _norm(s)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq[:max_candidates]


def _resolve_picks(picks, candidates: list[str]) -> list[str]:
    """Map 1-based indices chosen by the model to the exact candidate sentences."""
    out: list[str] = []
    for p in picks or []:
        try:
            i = int(p) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(candidates) and candidates[i] not in out:
            out.append(candidates[i])
    return out


def step1_themes(client: Groq, model: str, sample: str, candidates: list[str]) -> dict:
    """Themes + top 3 + 3 verbatim quotes + the fee issue. JSON only.

    The model selects quotes by number from ``candidates`` (verbatim slices of
    the reviews), so quotes are exact by construction. ``step1_verify_quotes``
    still re-checks them in code.
    """
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(candidates)) or "(none)"
    prompt = f"""You are a product analyst for the Groww investing app.
Below are recent Google Play reviews, one per line, prefixed by star rating:

{sample}

Below is a numbered list of candidate sentences taken VERBATIM from the reviews:
{numbered}

Return a JSON object with exactly this shape:
{{
  "themes": [{{"name": "<short label>", "count": <int>, "description": "<one sentence>"}}],
  "top_3_themes": ["<name>", "<name>", "<name>"],
  "quote_picks": [<number>, <number>, <number>],
  "fee_issue": {{
    "fee_name": "<the specific fee/charge users complain about>",
    "confusion_summary": "<one or two sentences: what users misunderstand or dislike>",
    "evidence_quote_picks": [<number>]
  }}
}}
Rules:
- themes: at most {config.MAX_THEMES}, ordered by how often they appear.
- top_3_themes: exactly {config.TOP_N_THEMES} names copied from themes.
- quote_picks: exactly {config.N_QUOTES} DIFFERENT numbers from the candidate list,
  choosing the sentences that best represent the top themes.
- fee_issue.fee_name must name a specific fee actually mentioned in the reviews
  (e.g. brokerage, DP charge, AMC, exit load, STT) -- never invent one.
- evidence_quote_picks: 0-2 numbers from the candidate list about that fee.
- Use ONLY the numbers shown in the candidate list. Never write out a sentence."""
    data = call_groq_json(client, model, prompt)
    data["quotes"] = _resolve_picks(data.get("quote_picks", []), candidates)
    fee = data.setdefault("fee_issue", {})
    fee["evidence_quotes"] = _resolve_picks(fee.get("evidence_quote_picks", []), candidates)
    return data


def step1_verify_quotes(themes: dict, corpus: set[str]) -> tuple[dict, list[str], list[str]]:
    """Return (themes, verified_quotes, dropped_quotes). Unverified are removed."""
    quotes = [str(q) for q in themes.get("quotes", []) if q]
    evidence = [str(q) for q in themes.get("fee_issue", {}).get("evidence_quotes", []) if q]
    verified = [q for q in quotes if verify_quote(q, corpus)]
    dropped = [q for q in quotes if q not in verified]
    themes["quotes"] = verified
    themes["fee_issue"]["evidence_quotes"] = [q for q in evidence if verify_quote(q, corpus)]
    return themes, verified, dropped


def step1_reask_quotes(client: Groq, model: str, candidates: list[str],
                       corpus: set[str], have: list[str]) -> list[str]:
    """Ask once more for additional quote picks when fewer than N_QUOTES verified.

    The model may only return numbers from the candidate list; the resolved
    sentences are re-verified in code. Never returns unverified text.
    """
    need = config.N_QUOTES - len(have)
    if need <= 0:
        return []
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(candidates)) or "(none)"
    prompt = f"""Candidate sentences taken VERBATIM from Groww reviews:
{numbered}

Already selected: {json.dumps(have, ensure_ascii=False)}

Return JSON: {{"quote_picks": [<number>, ...]}}
Pick {need} more DIFFERENT number(s) from the list above that reflect the main
themes. Use only the numbers shown. Never write out a sentence."""
    resp = call_groq_json(client, model, prompt)
    picked = _resolve_picks(resp.get("quote_picks", []), candidates)
    good = [q for q in picked if verify_quote(q, corpus) and q not in have]
    return good[:need]


def step2_pulse(client: Groq, model: str, themes: dict, fee: dict) -> str:
    """Weekly pulse (<=250 words) from themes JSON + quotes + fee issue ONLY."""
    fee_block = json.dumps(fee, ensure_ascii=False)
    themes_block = json.dumps(
        {"themes": themes.get("themes", []), "top_3_themes": themes.get("top_3_themes", [])},
        ensure_ascii=False,
    )
    prompt = f"""You write an internal weekly product pulse for the Groww team.
Use ONLY the structured input below -- do not add facts, numbers or quotes
that are not present in it.

Top themes (JSON):
{themes_block}

Identified fee issue (JSON):
{fee_block}

Write the pulse as plain text with exactly these sections:
Top themes
- ... (the top 3 themes, one line each)
Supporting quotes
- "..." (up to {config.N_QUOTES} quotes from the JSON above, verbatim)
Key observation
<one or two sentences>
Action ideas
1. ...
2. ...
3. ...

Total length must be under {config.MAX_PULSE_WORDS} words."""
    resp = call_groq_json(client, model, prompt + "\nReturn JSON: {\"weekly_pulse\": \"...\"}")
    text = str(resp.get("weekly_pulse", "")).strip()

    if len(text.split()) > config.MAX_PULSE_WORDS:
        resp = call_groq_json(
            client, model,
            prompt + f"\nYour previous pulse was too long. Shorten it to under {config.PULSE_REGEN_WORDS} words."
                    "\nReturn JSON: {\"weekly_pulse\": \"...\"}",
        )
        text = str(resp.get("weekly_pulse", "")).strip()
    return text


def step3_fee(client: Groq, model: str, fee: dict, today: date) -> tuple[list[str], list[str], str]:
    """Customer-facing explainer. Returns (bullets, sources, last_checked_line).

    Sources are filtered in code against config.SOURCE_LINKS; the
    'Last checked' line is appended by code, never by the model.
    """
    allowed = ", ".join(config.SOURCE_LINKS)
    prompt = f"""You write a short customer-facing clarification for Groww app users.
This is about ONE identified issue only:

{json.dumps(fee, ensure_ascii=False)}

Return JSON:
{{"bullets": ["<one clear, neutral, facts-only bullet>", ...]}}
Rules:
- at most {config.MAX_FEE_BULLETS} bullets, each a plain sentence.
- Neutral, facts-only tone. No blame, no promises, no apologies on behalf of anyone.
- Never state a fee amount, percentage or figure unless it appears verbatim in
  the input above. Otherwise say "see the official pricing page".
- Do not include URLs in the bullets; they are added separately."""
    resp = call_groq_json(client, model, prompt)
    bullets = [str(b).strip() for b in resp.get("bullets", []) if str(b).strip()]
    bullets = bullets[: config.MAX_FEE_BULLETS]  # hard cap, enforced in code

    # Pick up to 2 allow-listed sources; never trust model-supplied URLs.
    sources = config.SOURCE_LINKS[:2]
    last_checked = f"Last checked: {today.strftime('%d %b %Y')}"
    return bullets, sources, last_checked


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------
def run_gates(themes: dict, pulse: str, bullets: list[str], sources: list[str],
              verified_quotes: list[str], dropped_quotes: list[str]) -> list[tuple[str, bool, str]]:
    """Every rule the spec requires before approval is enabled."""
    out: list[tuple[str, bool, str]] = []

    n_themes = len(themes.get("themes", []))
    out.append((f"At most {config.MAX_THEMES} themes", 0 < n_themes <= config.MAX_THEMES, f"{n_themes} themes"))

    top3 = themes.get("top_3_themes", []) or []
    out.append((f"Exactly {config.TOP_N_THEMES} top themes", len(top3) == config.TOP_N_THEMES, f"{len(top3)}"))

    out.append((f"{config.N_QUOTES} verbatim quotes verified",
                len(verified_quotes) >= config.N_QUOTES,
                f"{len(verified_quotes)} verified, {len(dropped_quotes)} dropped"))

    words = len(pulse.split())
    out.append((f"Weekly pulse <= {config.MAX_PULSE_WORDS} words", 0 < words <= config.MAX_PULSE_WORDS, f"{words} words"))

    out.append((f"Fee explainer <= {config.MAX_FEE_BULLETS} bullets", 0 < len(bullets) <= config.MAX_FEE_BULLETS, f"{len(bullets)} bullets"))

    allowed = set(config.SOURCE_LINKS)
    out.append(("All sources on the allow-list", all(s in allowed for s in sources), f"{len(sources)} sources"))

    fee = themes.get("fee_issue", {})
    out.append(("Specific fee issue identified", bool(fee.get("fee_name")), fee.get("fee_name", "(none)")))
    return out


# ---------------------------------------------------------------------------
# MCP execution -- the ONLY place the MCP server is contacted.
# Invoked ONLY from the Approve & Execute button.
# ---------------------------------------------------------------------------
def run_mcp_actions(notes_payload: dict, email_payload: dict) -> list[dict]:
    """Call the two MCP tools over stdio. Returns parsed tool results."""

    async def _run() -> list[dict]:
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(config.BASE_DIR / config.MCP_SERVER_SCRIPT)],
            cwd=str(config.BASE_DIR),
        )
        results = []
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for name, args in (("append_to_notes", notes_payload),
                                   ("create_email_draft", email_payload)):
                    r = await session.call_tool(name, args)
                    raw = r.content[0].text if r.content else "{}"
                    try:
                        results.append({"tool": name, **json.loads(raw)})
                    except json.JSONDecodeError:
                        results.append({"tool": name, "status": "ok", "message": raw})
        return results

    return asyncio.run(_run())


def build_pending(themes: dict, pulse: str, bullets: list[str], sources: list[str],
                  newest: pd.Timestamp | None) -> tuple[dict, dict]:
    """Exact payloads that will be sent to the MCP tools after approval."""
    day = (newest.date() if newest is not None else date.today()).isoformat()
    notes = {
        "date": day,
        "top_themes": [f"{t.get('name')}: {t.get('description')}" for t in themes.get("themes", [])[: config.TOP_N_THEMES]],
        "weekly_pulse": pulse,
        "identified_fee_issue": themes.get("fee_issue", {}).get("fee_name", ""),
        "explanation_bullets": bullets,
        "source_links": sources,
    }
    body_lines = ["Weekly product pulse", "", pulse, "", "Support snippet (reusable)", ""]
    body_lines += [f"- {b}" for b in bullets]
    body_lines += [""] + [f"- {s}" for s in sources]
    body_lines += ["", f"Last checked: {date.today().strftime('%d %b %Y')}"]
    email = {
        "subject": f"Weekly Product Pulse + Customer Clarification \u2014 {day}",
        "body": "\n".join(body_lines),
    }
    return notes, email


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Groww Review Pulse", layout="wide")
    st.title("Groww Weekly Product Pulse & Fee Explainer")

    client = get_groq_client()
    if client is None:
        st.error("API key missing or invalid. Set `API_KEY` in `.streamlit/secrets.toml` "
                 "or in the environment (see README).")

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Groq model", config.GROQ_MODEL)
        weeks = st.slider("Analysis window (weeks)", config.WINDOW_WEEKS_MIN,
                          config.WINDOW_WEEKS_MAX, config.WINDOW_WEEKS_DEFAULT)

    uploaded = st.file_uploader("Upload reviews CSV (optional -- uses bundled data/reviews.csv if empty)",
                                type="csv")

    try:
        raw = load_raw(uploaded.getvalue() if uploaded is not None else None)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

    # ------------------------------------------------- column mapping (Step 0)
    cols = list(raw.columns)
    det_text = detect_column(cols, config.TEXT_COLUMN_CANDIDATES) or cols[0]
    det_rating = detect_column(cols, config.RATING_COLUMN_CANDIDATES) or cols[0]
    det_date = detect_column(cols, config.DATE_COLUMN_CANDIDATES)

    with st.sidebar:
        st.header("Column mapping")
        text_col = st.selectbox("Review text column", cols, index=cols.index(det_text))
        rating_col = st.selectbox("Rating column", cols, index=cols.index(det_rating))
        date_options = ["(none -- skip date filter)"] + cols
        date_default = date_options.index(det_date) if det_date else 0
        date_choice = st.selectbox("Date column", date_options, index=date_default)
        date_col = None if date_choice.startswith("(none)") else date_choice

    st.caption(f"Auto-detected mapping: text=`{text_col}`, rating=`{rating_col}`, date=`{date_col or 'none'}`")

    df = prepare(raw, text_col, rating_col, date_col)

    if df.empty:
        st.warning("No usable reviews after cleaning. Check the column mapping.")
        st.stop()

    win, newest, used_weeks = window_data(df, weeks)

    # ------------------------------------------------------------- metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews (cleaned)", len(df))
    c2.metric(f"Window: last {used_weeks} weeks", len(win))
    c3.metric("Newest review", newest.date().isoformat() if newest is not None else "n/a")
    c4.metric("Avg rating (window)", f"{win['rating'].mean():.2f}" if len(win) else "n/a")

    if date_col is None:
        st.info("No usable date column -- the week-window filter is skipped and all "
                "cleaned reviews are analysed.")
    elif used_weeks > weeks:
        st.info(f"Window widened to {used_weeks} weeks to reach at least "
                f"{config.MIN_REVIEWS_IN_WINDOW} reviews.")

    if win.empty:
        st.warning("No reviews inside the analysis window.")
        st.stop()

    # ------------------------------------------- Step 0 evidence + token gauge
    st.subheader("Step 0 \u2014 keyword frequency (no LLM)")
    kw = keyword_table(win)
    nonzero = kw[kw["reviews mentioning"] > 0].sort_values("reviews mentioning", ascending=False)
    st.dataframe(nonzero if len(nonzero) else kw, use_container_width=True, hide_index=True)

    sample_text, sample_texts, n_sent, n_total, tokens = build_sample(win)
    st.caption(f"LLM call 1 will receive {n_sent} of {n_total} reviews "
               f"(~{tokens} tokens; hard cap {config.MAX_LLM_REVIEWS}).")

    if client is None:
        st.stop()

    # ------------------------------------------------------------- analyse run
    if st.button("Analyze", type="primary"):
        corpus = {_norm(t) for t in sample_texts}  # exact strings the model saw
        candidates = build_quote_candidates(sample_texts)
        dropped: list[str] = []
        try:
            with st.spinner("Step 1/3: themes, quotes, fee issue..."):
                themes = step1_themes(client, model, sample_text, candidates)
                themes, verified, dropped = step1_verify_quotes(themes, corpus)
                if len(verified) < config.N_QUOTES:
                    with st.spinner("Step 1b/3: re-asking for verbatim quotes..."):
                        extra = step1_reask_quotes(client, model, candidates, corpus, verified)
                    verified = verified + extra
                    themes["quotes"] = verified
            if dropped:
                st.warning(f"{len(dropped)} quote(s) could not be verified against the "
                           "source reviews and were dropped (never displayed).")

            with st.spinner("Step 2/3: weekly pulse..."):
                pulse = step2_pulse(client, model, themes, themes.get("fee_issue", {}))

            with st.spinner("Step 3/3: customer fee explainer..."):
                bullets, sources, last_checked = step3_fee(client, model, themes.get("fee_issue", {}), date.today())

            st.session_state["result"] = {
                "themes": themes, "pulse": pulse, "bullets": bullets,
                "sources": sources, "last_checked": last_checked,
                "verified": verified, "dropped": dropped,
                "newest": newest,
            }
            st.session_state["executed"] = None
        except Exception as exc:  # friendly message, never a stack trace
            st.error(f"Analysis failed: {exc}")

    if "result" not in st.session_state or st.session_state["result"] is None:
        st.info("Set the options, then click **Analyze**.")
        st.stop()

    res = st.session_state["result"]
    themes, pulse = res["themes"], res["pulse"]
    bullets, sources = res["bullets"], res["sources"]
    newest = res["newest"]

    # ---------------------------------------------------------------- results
    st.subheader("Themes")
    for t in themes.get("themes", [])[: config.MAX_THEMES]:
        st.markdown(f"**{t.get('name')}** \u00b7 {t.get('count', '?')} mentions \u2014 {t.get('description', '')}")
    top3 = themes.get("top_3_themes", []) or []
    st.caption("Top 3: " + ", ".join(top3))

    if res["verified"]:
        st.subheader("Supporting quotes (verbatim, verified)")
        for q in res["verified"]:
            st.markdown(f"> \u201c{q}\u201d")

    st.subheader("Weekly product pulse")
    st.write(pulse)
    st.caption(f"Word count: {len(pulse.split())} / {config.MAX_PULSE_WORDS}")

    st.subheader("Customer-facing fee explainer")
    fee = themes.get("fee_issue", {})
    st.markdown(f"**Identified fee issue:** {fee.get('fee_name', '(none)')} \u2014 {fee.get('confusion_summary', '')}")
    for b in bullets:
        st.markdown(f"- {b}")
    for s in sources:
        st.markdown(f"- {s}")
    st.caption(res["last_checked"])

    # ------------------------------------------------------- approval gate
    st.subheader("Pending actions (nothing has been written yet)")
    gates = run_gates(themes, pulse, bullets, sources, res["verified"], res["dropped"])
    st.markdown("#### Validation gates")
    for name, ok, detail in gates:
        st.write(f"{'PASS' if ok else 'FAIL'} \u2014 {name} ({detail})")
    all_ok = all(ok for _, ok, _ in gates)

    notes_payload, email_payload = build_pending(themes, pulse, bullets, sources, newest)

    st.markdown("#### Exact payloads that will be sent to the MCP tools")
    st.code(json.dumps({"append_to_notes": notes_payload}, indent=2, ensure_ascii=False), language="json")
    st.code(json.dumps({"create_email_draft": email_payload}, indent=2, ensure_ascii=False), language="json")

    col_a, col_b = st.columns(2)
    approve = col_a.button("Approve & Execute", type="primary", disabled=not all_ok,
                           help="Calls the local MCP server: appends Notes.md, writes an .eml draft.")
    reject = col_b.button("Reject", help="Discard the pending results. Nothing is written.")

    if not all_ok:
        st.error("Approval disabled: one or more validation gates failed.")

    if reject:
        st.session_state["result"] = None
        st.session_state["executed"] = None
        st.info("Rejected. Pending results cleared; nothing was written to disk.")
        st.stop()

    if approve:
        try:
            with st.spinner("Running MCP actions..."):
                st.session_state["executed"] = run_mcp_actions(notes_payload, email_payload)
        except Exception as exc:
            st.error(f"MCP execution failed: {exc}")

    executed = st.session_state.get("executed")
    if executed:
        st.subheader("Executed")
        notes_path = config.NOTES_FILE
        draft_path = None
        for item in executed:
            if item.get("tool") == "append_to_notes":
                st.success(item.get("message", "Notes.md updated."))
            else:
                draft_path = item.get("path")
                st.success(item.get("message", "Email draft written."))

        if notes_path.exists():
            st.markdown("#### Saved Notes.md entry")
            st.code(notes_path.read_text(encoding="utf-8"), language="markdown")
            st.download_button("Download Notes.md", notes_path.read_bytes(),
                               file_name="Notes.md", mime="text/markdown")

        if draft_path and Path(draft_path).exists():
            st.markdown("#### Email draft (.eml)")
            st.code(Path(draft_path).read_text(encoding="utf-8"))
            st.download_button("Download email draft", Path(draft_path).read_bytes(),
                               file_name=Path(draft_path).name, mime="message/rfc822")


if __name__ == "__main__":
    main()
