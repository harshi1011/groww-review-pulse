# Groww Weekly Product Pulse & Fee Explainer

A local Streamlit app plus a local [MCP](https://modelcontextprotocol.io) (stdio)
server that turn scraped Google-Play reviews of the Groww app into (1) an
internal weekly product pulse and (2) a customer-facing fee explainer. After a
human clicks **Approve & Execute**, the app calls two MCP tools that append the
digest to `Notes.md` and write an `.eml` email draft. Nothing is ever emailed.

## 1. What it does

- Analyses the newest 8–12 weeks of reviews (relative to the newest review
  date in the CSV, not today) with three Groq LLM calls: themes + verbatim
  quotes + fee issue, weekly pulse, customer fee explainer.
- Blocks approval until validation gates pass (word/bullet limits, quotes
  verified verbatim against the source reviews, URLs on an allow-list).
- Only after explicit approval writes `Notes.md` and `drafts/<date>.eml` via
  the local MCP server.

## 2. Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
# edit .streamlit\secrets.toml and put your Groq key (https://console.groq.com/keys):
#   API_KEY = "gsk_..."

streamlit run app.py
```

The key is read from `st.secrets["API_KEY"]` (or the `API_KEY` / `GROQ_API_KEY`
environment variables) and is never hardcoded, printed or logged.

## 3. Where the MCP approval happens

All output is rendered in the browser first. The final section of the page
shows the exact JSON payloads that will be sent to the two MCP tools, plus
**Approve & Execute** and **Reject** buttons.

- `run_mcp_actions()` in `app.py` is the *only* place the MCP server is
  contacted, and it is unreachable except from the Approve button handler.
- `mcp_server.py` (stdio) exposes `append_to_notes` and `create_email_draft`.
  It only appends to `Notes.md` and writes `drafts/<date>.eml` — no SMTP,
  no OAuth, no network calls.
- **Reject** clears the pending results; nothing is written to disk.
- `Notes.md` and `drafts/` are gitignored and created only after approval.

## 4. Identified fee issue

The "identified fee issue" (fee name, confusion summary, evidence quotes) is
produced by LLM call 1 from the reviews themselves — it is not hardcoded.
Review it in the UI before approving; edit `config.SOURCE_LINKS` to control
which official Groww URLs may back the customer-facing explanation.

## 5. Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (exclude `.streamlit/secrets.toml`
   — it is already in `.gitignore`).
2. Open <https://share.streamlit.io> → **New app** → pick the repo, branch
   `main`, main file `app.py`.
3. **Advanced settings → Secrets** and paste:

   ```toml
   API_KEY = "gsk_..."
   ```

4. Deploy. The app runs against the repo's `data/reviews.csv`; upload another
   CSV in the UI to override it.

## 6. Sources

Official Groww pages allowed for the customer-facing explainer (edit in
`config.py` → `SOURCE_LINKS`):

- https://groww.in/pricing
- https://groww.in/help
- https://groww.in/blog
- https://groww.in/markets
