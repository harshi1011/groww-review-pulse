"""Central configuration for the Groww review-analysis app.

No secrets live here. The API key is read at runtime from
``st.secrets["API_KEY"]`` (falling back to the ``API_KEY`` environment
variable) and is never hardcoded, printed or logged.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (always relative to this file, so the app works from any CWD)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REVIEWS_CSV = DATA_DIR / "reviews.csv"          # bundled default input
SAMPLE_CSV = DATA_DIR / "reviews_sample.csv"    # 20 rows for the repo

NOTES_FILE = BASE_DIR / "Notes.md"
DRAFTS_DIR = BASE_DIR / "drafts"

MCP_SERVER_SCRIPT = "mcp_server.py"

# ---------------------------------------------------------------------------
# Groq / LLM
# ---------------------------------------------------------------------------
GROQ_MODEL = "qwen/qwen3.8-27b"   # overridable in the sidebar
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 3000
MAX_LLM_REVIEWS = 150        # hard cap on reviews sent to call 1
REVIEW_TRUNC_CHARS = 200     # each review is truncated to this many chars

# ---------------------------------------------------------------------------
# Analysis window (weeks, relative to the NEWEST review date in the file)
# ---------------------------------------------------------------------------
WINDOW_WEEKS_DEFAULT = 10
WINDOW_WEEKS_MIN = 8
WINDOW_WEEKS_MAX = 12
WINDOW_WEEKS_HARD_MAX = 16   # auto-widening stops here
MIN_REVIEWS_IN_WINDOW = 60   # widen by 1 week at a time until this is met

MIN_REVIEW_CHARS = 15        # drop empty / trivially short / duplicate reviews

# ---------------------------------------------------------------------------
# Output limits (enforced by validation gates)
# ---------------------------------------------------------------------------
MAX_THEMES = 5
TOP_N_THEMES = 3
N_QUOTES = 3
MAX_PULSE_WORDS = 250
PULSE_REGEN_WORDS = 230      # regeneration asks for this target
MAX_FEE_BULLETS = 6

# ---------------------------------------------------------------------------
# Column auto-detection candidates (first match wins; UI can override)
# ---------------------------------------------------------------------------
TEXT_COLUMN_CANDIDATES = ["text", "review", "content", "body", "reviewText"]
RATING_COLUMN_CANDIDATES = ["score", "rating", "appRating", "stars"]
DATE_COLUMN_CANDIDATES = ["reviewedAt", "reviewed_at", "at", "date", "reviewDate"]

# ---------------------------------------------------------------------------
# Keyword frequency table (Step 0, no LLM)
# ---------------------------------------------------------------------------
MONEY_KEYWORDS = [
    "charge", "fee", "brokerage", "deducted", "dp", "amc",
    "exit load", "hidden", "penalty", "stt", "margin", "refund",
]

# ---------------------------------------------------------------------------
# Source links -- the ONLY URLs the fee explainer may cite.
# Placeholder list: replace with the official Groww pages you want to cite.
# ---------------------------------------------------------------------------
SOURCE_LINKS = [
    "https://groww.in/pricing",
    "https://groww.in/help",
    "https://groww.in/blog",
    "https://groww.in/markets",
]
