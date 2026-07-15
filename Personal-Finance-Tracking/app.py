import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import pdfplumber
import plotly.graph_objects as go
import requests
import streamlit as st
import markdown as md_lib
from dotenv import load_dotenv
from supabase import create_client, Client
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# CrewAI's LLM() forwards to litellm under the hood. For "openrouter/..."
# model strings, litellm's own key-resolution order checks the
# OPENROUTER_API_KEY env var — not just the api_key= kwarg passed to LLM().
# In some litellm/crewai versions that kwarg doesn't reliably reach the
# request, so mirroring API_TOKEN into OPENROUTER_API_KEY here closes that
# gap without requiring a second, duplicate .env entry.
# ---------------------------------------------------------------------------
if os.getenv("API_TOKEN") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("API_TOKEN")

# ---------------------------------------------------------------------------
# Email (Gmail SMTP) — used to send the AI budget plan to the user after
# analysis. SMTP_APP_PASSWORD assumes a Gmail "app password", not the
# regular account password (Gmail blocks plain-password SMTP login).
# ---------------------------------------------------------------------------
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")


def render_plan_markdown(text: str) -> str:
    """
    Converts the AI's Markdown output (###, **bold**, - lists, --- rules)
    into real HTML. Used for both the in-app panel and the emailed copy so
    the plan reads as formatted text everywhere instead of raw ### / **
    markup — a single shared renderer keeps the two outputs in sync.
    """
    return md_lib.markdown(text, extensions=["extra", "sane_lists"])


def send_plan_email(to_email: str, username: str, plan_text: str) -> tuple[bool, str]:
    """Emails the generated budget plan to the signed-in user.
    Returns (success, error_message)."""
    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        return False, "Email isn't configured (missing SMTP_EMAIL / SMTP_APP_PASSWORD in .env)."
    if not to_email:
        return False, "No email address on this account."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your TR🎯CK-it budget plan"
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email

        msg.attach(MIMEText(plan_text, "plain"))

        plan_html = render_plan_markdown(plan_text)
        html_body = f"""
        <html>
        <head>
        <style>
            body {{ font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
                    background: #FAFAF7; color: #14151A; margin: 0; padding: 0; }}
            .trk-email-wrap {{ max-width: 600px; margin: 0 auto; padding: 28px 20px; }}
            .trk-email-logo {{ font-size: 22px; font-weight: 700; margin-bottom: 18px; }}
            .trk-email-plan {{ background: #14151A; color: #E6E6E0; border-radius: 18px;
                                padding: 24px 26px; line-height: 1.65; font-size: 14px; }}
            .trk-email-plan h1, .trk-email-plan h2, .trk-email-plan h3,
            .trk-email-plan h4 {{ color: #D6FF3F; margin: 18px 0 8px; }}
            .trk-email-plan h1:first-child, .trk-email-plan h2:first-child,
            .trk-email-plan h3:first-child, .trk-email-plan h4:first-child {{ margin-top: 0; }}
            .trk-email-plan strong {{ color: #ffffff; }}
            .trk-email-plan ul, .trk-email-plan ol {{ padding-left: 20px; margin: 8px 0; }}
            .trk-email-plan li {{ margin-bottom: 4px; }}
            .trk-email-plan hr {{ border: none; border-top: 1px solid #33342E; margin: 18px 0; }}
            .trk-email-plan p {{ margin: 10px 0; }}
            .trk-email-footer {{ color: #6B6F76; font-size: 12px; margin-top: 18px; }}
        </style>
        </head>
        <body>
            <div class="trk-email-wrap">
                <div class="trk-email-logo">TR🎯CK-it</div>
                <p>Hi {username},</p>
                <p>Here's the budget plan TR🎯CK-it just generated for you:</p>
                <div class="trk-email-plan">{plan_html}</div>
                <div class="trk-email-footer">— TR🎯CK-it</div>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)



# ---------------------------------------------------------------------------
# st.set_page_config MUST be the very first Streamlit command in the script.
# In the Supabase snippet it was called at the bottom, after st.title/tabs/
# etc. had already run — Streamlit throws a StreamlitAPIException the moment
# that happens, so it has to move up here, before anything else.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TR🎯CK-it", page_icon="🎯", layout="wide")

def html_block(content: str) -> None:
    """
    Renders an HTML string via st.markdown, safely.

    Multi-line f-strings built inside indented Python blocks carry that
    indentation into the string itself. Markdown treats lines indented 4+
    spaces as a preformatted code block, which silently breaks any HTML
    written this way. This collapses all line-boundary whitespace down to
    a single space before rendering, so indentation in the source can
    never leak into the rendered output.
    """
    normalized = re.sub(r"\s*\n\s*", " ", content).strip()
    st.markdown(normalized, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
LEMON = "#D6FF3F"
LEMON_DEEP = "#A8CC00"
INK = "#14151A"
MUTED = "#6B6F76"
BG = "#FAFAF7"
CARD = "#FFFFFF"
BORDER = "#ECECE4"
CORAL = "#E5484D"

CATEGORY_ICON = {
    "Transport": "🚗",
    "Dining": "🍔",
    "Groceries": "🛒",
    "Other": "💳",
}


# ---------------------------------------------------------------------------
# USD conversion
#
# CBN's own exchange rate page (cbn.gov.ng/rates/ExchRateByCurrency.html)
# renders its rate table via JavaScript after page load — a plain HTTP
# request (which is all a script can make) receives an empty table with no
# actual rate data, and CBN doesn't publish a documented public API. Rather
# than build a scraper against a page with no stable, fetchable data
# source, this uses Frankfurter (api.frankfurter.dev) — a free, no-key
# exchange-rate service that includes NGN, sourced from 10 blended
# providers, with full daily history back to 1999. It's the practical
# equivalent of "the day's official rate" without a fragile scraper that
# breaks the moment CBN's page markup changes.
# ---------------------------------------------------------------------------
def validate_openrouter_key(token: str) -> tuple[bool, str]:
    """
    Hits OpenRouter's own /auth/key endpoint directly with `requests`,
    bypassing crewai/litellm entirely. This isolates whether a failure is
    because the key itself is bad, versus a plumbing issue in how
    crewai/litellm forwards the key to OpenRouter's chat endpoint.
    Returns (is_valid, detail_message).
    """
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        if resp.status_code == 200:
            return True, "Key is valid and active on OpenRouter."
        return False, f"HTTP {resp.status_code}: {resp.text[:300]}"
    except Exception as e:
        return False, f"Couldn't reach OpenRouter to validate the key: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_usd_ngn_rate() -> tuple[float | None, str | None]:
    """
    Returns (rate, error_message): how many NGN equal 1 USD, using
    Frankfurter's dedicated /v2/rate/{base}/{quote} endpoint. Unlike
    /v2/rates?date=..., this always resolves to the latest *available*
    rate rather than a specific calendar date — Frankfurter publishes
    once a day, so pinning to "today" can come back empty for hours
    until that day's rate lands. Cached for an hour so toggling the
    switch repeatedly doesn't refetch every rerun.
    """
    try:
        resp = requests.get(
            "https://api.frankfurter.dev/v2/rate/USD/NGN",
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = data.get("rate")
        if rate is None:
            return None, "No USD/NGN rate in the response."
        return float(rate), None
    except Exception as e:
        return None, str(e)


def fmt_money(amount: float, fx_rate: float | None) -> str:
    """Formats an amount in ₦, or converts to $ if an FX rate is active."""
    if fx_rate:
        return f"${amount / fx_rate:,.2f}"
    return f"₦{amount:,.0f}"


CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {INK};
    }}

    .stApp {{
        background-color: {BG};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {CARD};
        border-right: 1px solid {BORDER};
    }}

    .trk-header {{
        display: flex;
        align-items: baseline;
        gap: 14px;
        margin-bottom: 4px;
    }}
    .trk-logo {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 34px;
        letter-spacing: -0.5px;
    }}
    .trk-tagline {{
        color: {MUTED};
        font-size: 15px;
        margin-bottom: 28px;
    }}

    .trk-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 4px 20px rgba(20,21,26,0.04);
        height: 100%;
    }}
    .trk-card-label {{
        color: {MUTED};
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .trk-card-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    .trk-card-sub {{
        color: {MUTED};
        font-size: 13px;
        margin-top: 4px;
    }}

    .trk-cat-row {{
        padding: 10px 0;
        border-bottom: 1px solid {BORDER};
    }}
    .trk-cat-row:last-child {{ border-bottom: none; }}
    .trk-cat-left {{ display: flex; align-items: center; gap: 10px; }}
    .trk-cat-icon {{
        width: 34px; height: 34px;
        border-radius: 10px;
        background: {LEMON};
        display: flex; align-items: center; justify-content: center;
        font-size: 16px;
    }}
    .trk-cat-name {{ font-weight: 600; font-size: 14px; }}
    .trk-cat-amount {{ font-weight: 600; font-size: 14px; }}
    .trk-bar-track {{
        background: #F1F1EA;
        border-radius: 6px;
        height: 6px;
        margin-top: 6px;
        width: 100%;
    }}
    .trk-bar-fill {{
        background: {LEMON_DEEP};
        border-radius: 6px;
        height: 6px;
    }}

    .trk-tx-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid {BORDER};
    }}
    .trk-tx-row:last-child {{ border-bottom: none; }}
    .trk-tx-left {{ display: flex; align-items: center; gap: 12px; }}
    .trk-tx-icon {{
        width: 38px; height: 38px;
        border-radius: 50%;
        background: {BG};
        border: 1px solid {BORDER};
        display: flex; align-items: center; justify-content: center;
        font-size: 17px;
    }}
    .trk-tx-desc {{ font-weight: 600; font-size: 14px; }}
    .trk-tx-cat {{ color: {MUTED}; font-size: 12px; }}
    .trk-tx-amount {{ color: {CORAL}; font-weight: 600; font-size: 14px; }}

    .trk-ai-panel {{
        background: {INK};
        color: white;
        border-radius: 18px;
        padding: 22px 24px;
    }}
    .trk-ai-panel h4 {{ margin-top: 0; color: {LEMON}; }}
    .trk-ai-panel .trk-ai-body {{
        color: #E6E6E0;
        font-size: 14px;
        line-height: 1.6;
    }}
    .trk-ai-body h1, .trk-ai-body h2, .trk-ai-body h3, .trk-ai-body h4,
    .trk-ai-body h5, .trk-ai-body h6 {{
        color: {LEMON};
        font-family: 'Space Grotesk', sans-serif;
        margin: 18px 0 8px;
    }}
    .trk-ai-body h1:first-child, .trk-ai-body h2:first-child,
    .trk-ai-body h3:first-child, .trk-ai-body h4:first-child {{ margin-top: 0; }}
    .trk-ai-body p {{ margin: 10px 0; }}
    .trk-ai-body strong {{ color: #ffffff; }}
    .trk-ai-body ul, .trk-ai-body ol {{ padding-left: 20px; margin: 8px 0; }}
    .trk-ai-body li {{ margin-bottom: 4px; }}
    .trk-ai-body hr {{ border: none; border-top: 1px solid #33342E; margin: 18px 0; }}
    .trk-ai-body code {{
        background: #24261F; color: {LEMON};
        padding: 2px 6px; border-radius: 6px; font-size: 13px;
    }}
    .trk-ai-body table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    .trk-ai-body th, .trk-ai-body td {{
        border: 1px solid #33342E; padding: 6px 10px; text-align: left;
    }}

    div.st-key-target_card {{
        background: {CARD};
        border: 1px solid {BORDER} !important;
        border-radius: 18px !important;
        padding: 22px 24px !important;
        box-shadow: 0 4px 20px rgba(20,21,26,0.04);
    }}

    div.st-key-trend_card {{
        background: {CARD};
        border: 1px solid {BORDER} !important;
        border-radius: 18px !important;
        padding: 22px 24px !important;
        box-shadow: 0 4px 20px rgba(20,21,26,0.04);
    }}

    div.st-key-fx_card {{
        background: {CARD};
        border: 1px solid {BORDER} !important;
        border-radius: 18px !important;
        padding: 22px 24px !important;
        box-shadow: 0 4px 20px rgba(20,21,26,0.04);
    }}
    div.st-key-fx_card div[data-testid="stToggle"] label p {{
        font-size: 12px !important;
        font-weight: 600 !important;
        color: {MUTED} !important;
    }}

    div.st-key-auth_card {{
        background: {CARD};
        border: 1px solid {BORDER} !important;
        border-radius: 18px !important;
        padding: 28px 28px 8px !important;
        box-shadow: 0 4px 20px rgba(20,21,26,0.04);
        margin-top: 24px;
    }}

    /* Sign in / Sign up tabs, styled to match the lemon accent */
    button[data-baseweb="tab"] {{
        font-weight: 600;
        color: {MUTED};
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {INK} !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {LEMON_DEEP} !important;
    }}
    div[data-baseweb="tab-border"] {{
        background-color: {BORDER} !important;
    }}

    /* Text inputs, matched to the card/border/lemon-focus styling elsewhere */
    div[data-testid="stTextInput"] input {{
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
        background: {BG} !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border-color: {LEMON_DEEP} !important;
        box-shadow: 0 0 0 1px {LEMON_DEEP} !important;
    }}

    /* Pill-style toggle for the Day / Month / Year radio */
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        gap: 6px;
    }}
    div[data-testid="stRadio"] label {{
        background: #F1F1EA;
        border-radius: 999px;
        padding: 4px 14px;
        margin: 0 !important;
    }}
    div[data-testid="stRadio"] label > div:first-child {{
        display: none;
    }}
    div[data-testid="stRadio"] label:has(input:checked) {{
        background: {LEMON};
    }}
    div[data-testid="stRadio"] label p {{
        font-size: 13px !important;
        font-weight: 600 !important;
        color: {INK} !important;
    }}

    div.stButton > button {{
        background-color: {LEMON};
        color: {INK};
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6em 1em;
    }}
    div.stButton > button:hover {{
        background-color: {LEMON_DEEP};
        color: {INK};
    }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Supabase auth setup
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

for key, default in {
    "authenticated": False,
    "user": None,
    "username": "",
    "email": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def sign_up_user(username: str, email: str, password: str):
    """Register a new user via Supabase Auth, storing username in metadata."""
    try:
        return supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"username": username}},
        })
    except Exception as e:
        st.error(f"Signup failed: {e}")
        return None


def sign_in_user(email: str, password: str):
    try:
        return supabase.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None


def logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.username = ""
    st.session_state.email = ""
    st.rerun()


def get_current_user():
    try:
        return supabase.auth.get_user().user
    except Exception:
        return None


def check_session() -> bool:
    user = get_current_user()
    if user:
        st.session_state.authenticated = True
        st.session_state.user = user
        st.session_state.email = user.email
        metadata = user.user_metadata or {}
        st.session_state.username = metadata.get("username", user.email.split("@")[0])
        return True
    return False


def reset_password(email: str):
    try:
        supabase.auth.reset_password_email(email)
        st.success("Password reset email sent.")
    except Exception as e:
        st.error(e)


def auth_page():
    html_block(f"""
        <div class="trk-header" style="margin-top: 40px; justify-content: center;">
            <span class="trk-logo">TR🎯CK-it</span>
        </div>
        <div class="trk-tagline" style="text-align: center;">
            Your AI personal finance assistant — sign in to continue.
        </div>
    """)

    left, mid, right = st.columns([1, 1.3, 1])
    with mid:
        with st.container(border=True, key="auth_card"):
            tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Sign Up"])

            with tab1:
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")

                if st.button("Sign In", use_container_width=True):
                    response = sign_in_user(email, password)
                    if response and response.user:
                        user = response.user
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.email = user.email
                        metadata = user.user_metadata or {}
                        st.session_state.username = metadata.get("username", user.email.split("@")[0])
                        st.success("Login successful!")
                        st.rerun()

                if st.button("Forgot password?", use_container_width=True):
                    if email:
                        reset_password(email)
                    else:
                        st.warning("Enter your email above first.")

            with tab2:
                username = st.text_input("Username", key="signup_username")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password")
                confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

                if st.button("Create Account", use_container_width=True):
                    if password != confirm:
                        st.error("Passwords do not match.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        response = sign_up_user(username, email, password)
                        if response:
                            st.success("Account created successfully!")
                            st.info("Please verify your email before logging in.")


# ---------------------------------------------------------------------------
# Auth gate — nothing below this point runs until the user is signed in.
# ---------------------------------------------------------------------------
if not st.session_state.authenticated:
    check_session()

if not st.session_state.authenticated:
    auth_page()
    st.stop()



# ---------------------------------------------------------------------------
# LLM (OpenRouter, gpt-4o-mini — paid, requires OpenRouter credit balance)
# ---------------------------------------------------------------------------
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_TOKEN"),
)


CATEGORY_KEYWORDS = {
    "uber": "Transport",
    "kfc": "Dining",
    "shoprite": "Groceries",
}


def categorize(description: str) -> str:
    desc = description.lower()
    for key, cat in CATEGORY_KEYWORDS.items():
        if key in desc:
            return cat
    return "Other"


# ---------------------------------------------------------------------------
# Universal file ingestion — normalizes CSV or PDF bank statements into the
# same shape: columns ['description', 'amount'] and optionally ['date'].
#
# Design choice: for both formats, only DEBIT (money-out / spending) rows
# are kept. Money-in (credits/deposits/salary) is intentionally discarded —
# the user provides take-home income and a savings target manually, so the
# app never needs to infer income from the transaction data itself.
# ---------------------------------------------------------------------------
HEADER_ALIASES = {
    "date": ["date", "trans date", "value date", "posting date", "txn date"],
    "description": ["narration", "description", "details", "particulars", "remarks", "transaction details"],
    "debit": ["debit", "withdrawal", " dr", "debit amount", "money out", "amount(dr)", "amount (dr)"],
    "credit": ["credit", "deposit", " cr", "credit amount", "money in", "amount(cr)", "amount (cr)"],
    "amount": ["amount", "value"],
}


def _match_header(col_name: str) -> str | None:
    c = f" {str(col_name).lower().strip()} "
    for key, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if alias in c:
                return key
    return None


def _clean_number(val) -> float | None:
    try:
        s = str(val).replace(",", "").replace("₦", "").replace("N", "").strip()
        if s in ("", "-", "nan", "none"):
            return None
        n = float(s)
        return abs(n) if n != 0 else None
    except (ValueError, TypeError):
        return None


def _load_csv(uploaded_file) -> tuple[pd.DataFrame, float]:
    """Returns (debit_only_transactions, total_money_in)."""
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]

    col_map = {}
    for col in df.columns:
        matched = _match_header(col)
        if matched and matched not in col_map:
            col_map[matched] = col

    desc_col = col_map.get("description") or ("description" if "description" in df.columns else None)
    if desc_col is None:
        raise ValueError(
            "CSV must have a description/narration column (e.g. 'description', "
            "'narration', or 'details')."
        )
    date_col = col_map.get("date") or ("date" if "date" in df.columns else None)

    if "debit" in col_map:
        rows = []
        total_in = 0.0
        for _, r in df.iterrows():
            desc = r.get(desc_col)
            if pd.isna(desc) or not str(desc).strip():
                continue
            if "credit" in col_map:
                credit_amt = _clean_number(r[col_map["credit"]])
                if credit_amt is not None:
                    total_in += credit_amt
                    continue
            amt = _clean_number(r[col_map["debit"]])
            if amt is None:
                continue
            rows.append({
                "description": str(desc).strip(),
                "amount": amt,
                "date": r.get(date_col) if date_col else None,
            })
        return pd.DataFrame(rows).reset_index(drop=True), total_in

    amount_col = col_map.get("amount") or ("value" if "value" in df.columns else None)
    if amount_col is None:
        raise ValueError(
            "CSV must have an 'amount'/'value' column, or separate "
            "'debit'/'credit' columns."
        )

    numeric = pd.to_numeric(df[amount_col], errors="coerce")
    has_negative = (numeric < 0).any()
    has_positive = (numeric > 0).any()

    work = df.copy()
    total_in = 0.0
    if has_negative and has_positive:
        total_in = float(numeric[numeric > 0].sum())
        mask = numeric < 0
        work = work.loc[mask].copy()
        work["amount"] = numeric.loc[mask].abs()
    else:
        work["amount"] = numeric.abs()

    work = work.rename(columns={desc_col: "description"})
    if date_col:
        work = work.rename(columns={date_col: "date"})

    keep_cols = [c for c in ["description", "amount", "date"] if c in work.columns]
    result = work[keep_cols].dropna(subset=["description", "amount"]).reset_index(drop=True)
    return result, total_in


def _extract_pdf_tables(uploaded_file) -> tuple[pd.DataFrame, float]:
    """Primary strategy: look for a transactions table with recognizable headers.
    Returns (debit_only_transactions, total_money_in)."""
    rows = []
    total_in = 0.0
    uploaded_file.seek(0)
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table or len(table) < 2:
                    continue
                header = table[0]
                col_map = {}
                for idx, col in enumerate(header):
                    if col is None:
                        continue
                    matched = _match_header(col)
                    if matched and matched not in col_map:
                        col_map[matched] = idx
                if "description" not in col_map:
                    continue

                for row in table[1:]:
                    if not row or col_map["description"] >= len(row):
                        continue
                    desc = row[col_map["description"]]
                    if not desc or not str(desc).strip():
                        continue

                    amount = None
                    if "debit" in col_map and col_map["debit"] < len(row):
                        amount = _clean_number(row[col_map["debit"]])
                    if amount is None and "credit" in col_map and col_map["credit"] < len(row):
                        credit_amt = _clean_number(row[col_map["credit"]])
                        if credit_amt is not None:
                            total_in += credit_amt
                            continue
                    if amount is None and "amount" in col_map and col_map["amount"] < len(row):
                        amount = _clean_number(row[col_map["amount"]])
                    if amount is None:
                        continue

                    date_val = None
                    if "date" in col_map and col_map["date"] < len(row):
                        date_val = row[col_map["date"]]

                    rows.append({
                        "description": str(desc).strip().replace("\n", " "),
                        "amount": amount,
                        "date": date_val,
                    })
    return pd.DataFrame(rows), total_in


_PDF_LINE_PATTERN = re.compile(
    r"^(?P<date>\d{1,2}[\/\-. ][A-Za-z0-9]{2,9}[\/\-. ]\d{2,4})\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<amount>[\d,]+\.\d{2})\s*$"
)


def _extract_pdf_text_fallback(uploaded_file) -> tuple[pd.DataFrame, float]:
    """Fallback for PDFs with no extractable table structure: regex over raw
    text lines. No debit/credit column signal is available in loose text,
    so every matched line is treated as spending (Money In stays 0 here —
    a documented limitation of this fallback path)."""
    rows = []
    uploaded_file.seek(0)
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                m = _PDF_LINE_PATTERN.match(line.strip())
                if not m:
                    continue
                amount = _clean_number(m.group("amount"))
                if amount is None:
                    continue
                rows.append({
                    "description": m.group("desc").strip(),
                    "amount": amount,
                    "date": m.group("date"),
                })
    return pd.DataFrame(rows), 0.0


def load_transactions(uploaded_file) -> tuple[pd.DataFrame, float]:
    """
    Reads an uploaded CSV or PDF and returns (transactions, total_money_in).
    Raises ValueError with a user-facing message if nothing could be parsed.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return _load_csv(uploaded_file)

    if name.endswith(".pdf"):
        df, total_in = _extract_pdf_tables(uploaded_file)
        if df.empty:
            df, total_in = _extract_pdf_text_fallback(uploaded_file)
        if df.empty:
            raise ValueError(
                "Couldn't find any transactions in this PDF. It may be a "
                "scanned/image-based statement (no extractable text) or use "
                "a layout this parser doesn't recognize yet."
            )
        return df.reset_index(drop=True), total_in

    raise ValueError("Unsupported file type. Please upload a CSV or PDF.")


# ---------------------------------------------------------------------------
# Deterministic waste detection — plain Python, not an LLM tool-call loop.
# ---------------------------------------------------------------------------
def detect_waste(df: pd.DataFrame, amount_col: str = "amount") -> list[str]:
    flags = []
    for _, row in df.iterrows():
        desc = str(row["description"])
        value = row[amount_col]
        if "subscription" in desc.lower() and value > 10000:
            flags.append(f"High subscription cost: {desc} (₦{value:,.0f})")
        if value > 50000:
            flags.append(f"Large transaction: {desc} (₦{value:,.0f})")
    return flags


def build_context_summary(summary: dict) -> str:
    """Plain-text digest of the already-computed numbers, handed to the LLM
    as context instead of making it re-derive them via tool calls."""
    lines = [
        f"Total spending: ₦{summary['total_spending']:,.0f} across "
        f"{summary['num_transactions']} transactions",
        f"Average transaction: ₦{summary['avg_transaction']:,.0f}",
        "",
        "Spending by category:",
    ]
    for cat, amt in summary["by_category"].items():
        lines.append(f"  - {cat}: ₦{amt:,.0f}")

    waste_flags = detect_waste(summary["df"], summary["amount_col"])
    lines.append("")
    if waste_flags:
        lines.append("Flagged potentially wasteful transactions:")
        for flag in waste_flags[:20]:
            lines.append(f"  - {flag}")
        if len(waste_flags) > 20:
            lines.append(f"  ...and {len(waste_flags) - 20} more not shown")
    else:
        lines.append("No wasteful spending patterns detected.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CrewAI: a single agent for the one part that genuinely benefits from an
# LLM — turning the numbers above into a written, personalized plan.
# ---------------------------------------------------------------------------
@tool("Budget Calculator")
def budget_calculator(income: float, savings_pct: float) -> dict:
    """
    Splits income using the user's own savings target percentage.
    Savings = income * savings_pct / 100. The remainder is split between
    Essentials and Wants using a 70/30 ratio.
    """
    savings_pct = max(0.0, min(savings_pct, 100.0))
    savings = income * (savings_pct / 100)
    remainder = income - savings
    return {
        "Savings": savings,
        "Essentials": remainder * 0.7,
        "Wants": remainder * 0.3,
    }


budget_agent = Agent(
    role="Budget Planner Agent",
    goal="Generate a personalized, encouraging budget plan with concrete savings tips",
    backstory=(
        "An experienced Nigerian personal finance coach who gives practical, "
        "specific advice grounded in the person's actual spending data — "
        "never generic filler."
    ),
    tools=[budget_calculator],
    llm=llm,
)


def build_crew(context_summary: str, monthly_income: float, savings_pct: float, savings_goal: str) -> Crew:
    task = Task(
        description=(
            "Here is a summary of the user's actual spending data for this "
            f"period:\n\n{context_summary}\n\n"
            f"The user's monthly take-home income is ₦{monthly_income:,.0f}. "
            f"They want to save {savings_pct}% of that income (not a generic "
            "50/30/20 rule — this exact percentage is what they chose). "
            f"Their savings goal is: {savings_goal}.\n\n"
            "Use the Budget Calculator tool with this exact income and "
            "savings_pct to get the Essentials/Wants/Savings split. Then "
            "write a short, encouraging, specific budget plan: comment on "
            "their actual category breakdown, call out any flagged "
            "wasteful spending by name, and give concrete savings tips "
            "tied to their stated goal."
        ),
        agent=budget_agent,
        expected_output=(
            "A personalized budget plan with savings tips, written in a "
            "warm, practical tone."
        ),
    )
    return Crew(agents=[budget_agent], tasks=[task])


# ---------------------------------------------------------------------------
# Deterministic summary
# ---------------------------------------------------------------------------
def compute_summary(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount", "description"])
    df["category"] = df["description"].apply(categorize)

    has_date = "date" in df.columns
    if has_date:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        if df["date"].notna().sum() == 0:
            has_date = False

    total_spending = df["amount"].sum()
    num_transactions = len(df)
    avg_transaction = df["amount"].mean() if num_transactions else 0
    by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    recent = df.sort_values("amount", ascending=False).head(8)

    return {
        "total_spending": total_spending,
        "num_transactions": num_transactions,
        "avg_transaction": avg_transaction,
        "by_category": by_category,
        "recent": recent,
        "amount_col": "amount",
        "has_date": has_date,
        "df": df,
    }


def spending_by_period(df: pd.DataFrame, amount_col: str, timeframe: str) -> pd.DataFrame:
    """Groups spending into Day / Month / Year buckets for the trend chart."""
    d = df.dropna(subset=["date"]).copy()
    if timeframe == "Day":
        d["period"] = d["date"].dt.floor("D")
    elif timeframe == "Month":
        d["period"] = d["date"].dt.to_period("M").dt.to_timestamp()
    else:
        d["period"] = d["date"].dt.to_period("Y").dt.to_timestamp()
    grouped = (
        d.groupby("period")[amount_col]
        .sum()
        .reset_index()
        .sort_values("period")
    )
    return grouped


def trend_chart(grouped: pd.DataFrame, amount_col: str, timeframe: str, fx_rate: float | None = None) -> go.Figure:
    tick_format = {"Day": "%b %d", "Month": "%b %Y", "Year": "%Y"}[timeframe]
    y_values = grouped[amount_col] / fx_rate if fx_rate else grouped[amount_col]
    hover_prefix = "$" if fx_rate else "₦"
    hover_fmt = ":,.2f" if fx_rate else ":,.0f"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped["period"],
        y=y_values,
        mode="lines+markers",
        line=dict(color=LEMON_DEEP, width=3, shape="spline"),
        marker=dict(color=LEMON_DEEP, size=6),
        fill="tozeroy",
        fillcolor="rgba(168,204,0,0.12)",
        hovertemplate=f"{hover_prefix}%{{y{hover_fmt}}}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, tickfont=dict(color=MUTED, size=11), tickformat=tick_format),
        yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(color=MUTED, size=11)),
    )
    return fig


def target_ring(percent_spent: float) -> go.Figure:
    """Signature element: a lemon target ring showing % of income spent."""
    percent_spent = max(0, min(percent_spent, 100))
    fig = go.Figure(go.Pie(
        values=[percent_spent, 100 - percent_spent],
        hole=0.72,
        marker=dict(colors=[LEMON_DEEP, "#F1F1EA"]),
        textinfo="none",
        sort=False,
        direction="clockwise",
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=180,
        annotations=[dict(
            text=f"<b>{percent_spent:.0f}%</b><br><span style='font-size:11px;color:{MUTED}'>of income</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=22, color=INK)
        )],
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 👋 Hi {st.session_state.username}")
    if st.button("Log out", use_container_width=True):
        logout()

    st.markdown("### 🎯 Your details")

    st.markdown("**Connect a data source**")
    data_source = st.radio(
        "Data source",
        ["File", "Bank account", "SQL database"],
        horizontal=True,
        label_visibility="collapsed",
        key="data_source",
    )

    uploaded_file = None
    if data_source == "File":
        uploaded_file = st.file_uploader(
            "Upload transactions (CSV or PDF)", type=["csv", "pdf"]
        )
    elif data_source == "Bank account":
        html_block("""
            <div class="trk-card" style="padding: 16px 18px;">
                <div class="trk-card-sub">
                    🏦 <b>Coming soon.</b> Securely connect a Nigerian bank
                    account via Mono for automatic, real-time transactions —
                    no CSV/PDF export needed.
                </div>
            </div>
        """)
    else:
        html_block("""
            <div class="trk-card" style="padding: 16px 18px;">
                <div class="trk-card-sub">
                    🗄️ <b>Coming soon.</b> Point TR🎯CK-it at your own
                    transactions database (e.g. Postgres/MySQL) and query it
                    directly instead of exporting a file.
                </div>
            </div>
        """)

    st.divider()
    monthly_income = st.number_input(
        "Monthly take-home income (₦)", min_value=0.0, step=1000.0, format="%.2f"
    )
    savings_pct = st.number_input(
        "Target savings (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0,
        help="What % of your income you want to save. The rest is split "
             "70/30 between essentials and wants."
    )
    savings_goal = st.text_input(
        "Savings goal (optional)", placeholder="e.g. emergency fund, vacation"
    )
    analyze_clicked = st.button("Analyze", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
html_block(f"""
    <div class="trk-header">
        <span class="trk-logo">TR🎯CK-it</span>
    </div>
    <div class="trk-tagline">
        Hi, {st.session_state.username} — see exactly where your money goes,
        then hit your target.
    </div>
""")

if uploaded_file is None:
    html_block("""
        <div class="trk-card">
            👋 Upload a transactions CSV or PDF bank statement from the sidebar
            to get started. CSVs need a <code>description</code> column and an
            <code>amount</code> (or <code>value</code>) column. PDF statements
            are parsed automatically — most bank layouts are supported.
        </div>
    """)
    st.stop()

try:
    raw_df, total_money_in = load_transactions(uploaded_file)
except ValueError as e:
    st.error(str(e))
    st.stop()

if raw_df.empty:
    st.error("No transactions could be found in this file.")
    st.stop()

summary = compute_summary(raw_df)

# ---------------------------------------------------------------------------
# Top row: KPI cards + target ring
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([1.1, 0.9, 0.9])

with col1:
    percent_spent = (
        (summary["total_spending"] / monthly_income * 100) if monthly_income else 0
    )
    with st.container(border=True, key="target_card"):
        html_block('<div class="trk-card-label">Spending vs. income target</div>')
        st.plotly_chart(target_ring(percent_spent), use_container_width=True, config={"displayModeBar": False})

with col2:
    with st.container(border=True, key="fx_card"):
        label_col, toggle_col = st.columns([1.6, 1])
        with label_col:
            html_block('<div class="trk-card-label">Total spending</div>')
        with toggle_col:
            show_usd = st.toggle("USD", value=False, key="fx_toggle")

        fx_rate = None
        if show_usd:
            fx_rate, fx_error = fetch_usd_ngn_rate()
            if fx_rate is None:
                st.caption(f"⚠️ Couldn't fetch a live rate — showing ₦ instead. ({fx_error})")

        html_block(f"""
            <div class="trk-card-value">{fmt_money(summary['total_spending'], fx_rate)}</div>
            <div class="trk-card-sub">{summary['num_transactions']} transactions</div>
        """)

with col3:
    net = monthly_income - summary["total_spending"] if monthly_income else None
    net_label = "—" if net is None else fmt_money(net, fx_rate)
    net_sub = "" if net is None else ("surplus" if net >= 0 else "deficit")
    html_block(f"""
        <div class="trk-card">
            <div class="trk-card-label">Avg. transaction</div>
            <div class="trk-card-value">{fmt_money(summary['avg_transaction'], fx_rate)}</div>
            <div class="trk-card-sub">{net_label} {net_sub}</div>
        </div>
    """)

st.write("")

# ---------------------------------------------------------------------------
# Second row: Money in vs. out card beside the spending-over-time trend chart
# ---------------------------------------------------------------------------
moneyin_col, trend_col = st.columns([1, 2])

with moneyin_col:
    html_block(f"""
        <div class="trk-card">
            <div class="trk-card-label">Money in vs. out</div>
            <div class="trk-card-sub" style="margin-top: 8px; font-size: 13px;">Money in</div>
            <div class="trk-card-value" style="font-size: 22px; color: {LEMON_DEEP};">
                {fmt_money(total_money_in, fx_rate)}
            </div>
            <div class="trk-card-sub" style="margin-top: 10px; font-size: 13px;">Money out</div>
            <div class="trk-card-value" style="font-size: 22px; color: {CORAL};">
                {fmt_money(summary['total_spending'], fx_rate)}
            </div>
        </div>
    """)

with trend_col:
    if summary["has_date"]:
        with st.container(border=True, key="trend_card"):
            html_block('<div class="trk-card-label">Spending over time</div>')
            timeframe = st.radio(
                "Timeframe",
                ["Day", "Month", "Year"],
                horizontal=True,
                label_visibility="collapsed",
                key="trend_timeframe",
            )
            grouped = spending_by_period(summary["df"], summary["amount_col"], timeframe)
            if grouped.empty:
                st.caption("Not enough dated transactions to show a trend.")
            else:
                st.plotly_chart(
                    trend_chart(grouped, summary["amount_col"], timeframe, fx_rate),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
    else:
        html_block("""
            <div class="trk-card">
                <div class="trk-card-label">Spending over time</div>
                <div class="trk-card-sub" style="margin-top: 10px;">
                    Add a <code>date</code> column to your CSV (e.g. 2026-06-15) to
                    see your spending trend by day, month, or year.
                </div>
            </div>
        """)

st.write("")

# ---------------------------------------------------------------------------
# Middle row: category breakdown + transactions list
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.3])

with col_left:
    max_val = summary["by_category"].max()
    rows_html = ""
    for cat, amt in summary["by_category"].items():
        pct = (amt / max_val * 100) if max_val else 0
        icon = CATEGORY_ICON.get(cat, "💳")
        rows_html += f"""
        <div class="trk-cat-row">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="trk-cat-left">
                    <div class="trk-cat-icon">{icon}</div>
                    <div class="trk-cat-name">{cat}</div>
                </div>
                <div class="trk-cat-amount">{fmt_money(amt, fx_rate)}</div>
            </div>
            <div class="trk-bar-track"><div class="trk-bar-fill" style="width:{pct}%;"></div></div>
        </div>
        """
    html_block(f"""
        <div class="trk-card">
            <div class="trk-card-label">Spending by category</div>
            {rows_html}
        </div>
    """)

with col_right:
    rows_html = ""
    for _, row in summary["recent"].iterrows():
        cat = row["category"]
        icon = CATEGORY_ICON.get(cat, "💳")
        rows_html += f"""
        <div class="trk-tx-row">
            <div class="trk-tx-left">
                <div class="trk-tx-icon">{icon}</div>
                <div>
                    <div class="trk-tx-desc">{row['description']}</div>
                    <div class="trk-tx-cat">{cat}</div>
                </div>
            </div>
            <div class="trk-tx-amount">-{fmt_money(row[summary['amount_col']], fx_rate)}</div>
        </div>
        """
    html_block(f"""
        <div class="trk-card">
            <div class="trk-card-label">Largest transactions</div>
            {rows_html}
        </div>
    """)

st.write("")

# ---------------------------------------------------------------------------
# AI Analysis panel
#
# Streamlit reruns the whole script on every interaction — including a
# click on a "Send to email" button placed after this panel. Without
# saving the generated plan somewhere that survives that rerun, clicking
# Send would land back here with analyze_clicked=False and lose the plan
# entirely. Stashing it in st.session_state is what lets the email button
# work as a separate, deliberate action instead of firing automatically.
# ---------------------------------------------------------------------------
html_block('<div class="trk-card-label" style="margin-bottom:10px;">🤖 AI analysis & budget plan</div>')

if analyze_clicked:
    if not monthly_income or monthly_income <= 0:
        st.error("Please enter a valid monthly income before analyzing.")
        st.stop()

    api_token = os.getenv("API_TOKEN")
    if not api_token:
        st.error(
            "No OpenRouter API key found. Set `API_TOKEN` in your `.env` file "
            "to a valid OpenRouter key (from https://openrouter.ai/keys)."
        )
        st.stop()

    with st.spinner("Checking OpenRouter key..."):
        key_valid, key_detail = validate_openrouter_key(api_token)

    if not key_valid:
        st.error(
            f"OpenRouter rejected `API_TOKEN` directly (confirmed against "
            f"their own /auth/key endpoint, independent of this app's code): "
            f"{key_detail}\n\nGenerate a fresh key at "
            f"https://openrouter.ai/keys and paste it into `.env` as "
            f"`API_TOKEN=...` with no quotes/spaces."
        )
        st.stop()

    with st.spinner("Running AI analysis..."):
        context_summary = build_context_summary(summary)
        try:
            crew = build_crew(
                context_summary,
                monthly_income,
                savings_pct,
                savings_goal or "general savings",
            )
            result = crew.kickoff()
        except Exception as e:
            err_text = str(e)
            if "402" in err_text or "Insufficient credits" in err_text:
                st.error(
                    "OpenRouter returned a 402 (\"Insufficient credits\"). "
                    "`gpt-4o-mini` is a paid model on OpenRouter and draws "
                    "down your account's credit balance — top up at "
                    "https://openrouter.ai/settings/credits, or switch the "
                    "model in `llm = LLM(...)` to a `:free`-suffixed model "
                    "(e.g. `openrouter/openai/gpt-oss-120b:free`) if you'd "
                    "rather not spend credits."
                )
            elif "401" in err_text or "User not found" in err_text:
                st.error(
                    "OpenRouter's chat endpoint returned a 401 (\"User not "
                    "found\") even though the key just validated successfully "
                    "against OpenRouter's own /auth/key endpoint. That points "
                    "to crewai/litellm not forwarding the key correctly — try "
                    "fully restarting the app (env vars are read once at "
                    "process start)."
                )
            elif "429" in err_text or "rate" in err_text.lower():
                st.error(
                    "OpenRouter rate-limited this request. Free-tier models "
                    "are capped (around 20 requests/minute and 200/day per "
                    "OpenRouter's published limits) — wait a bit and try "
                    "again, or add credits to lift the cap."
                )
            else:
                st.error(f"AI analysis failed: {err_text}")
            st.stop()

    # Save so the plan survives the rerun triggered by the email button below.
    st.session_state["plan_text"] = str(result)
    st.session_state["plan_email_sent"] = False

if "plan_text" not in st.session_state:
    html_block("""
        <div class="trk-ai-panel">
            <h4>Ready when you are</h4>
            <div class="trk-ai-body">
                Enter your income in the sidebar and click <b>Analyze</b> to get
                a personalized budget plan, waste alerts, and savings tips
                based on this data.
            </div>
        </div>
    """)
    st.stop()

plan_text = st.session_state["plan_text"]
result_html = render_plan_markdown(plan_text)
html_block(f"""
    <div class="trk-ai-panel">
        <h4>Your plan</h4>
        <div class="trk-ai-body">{result_html}</div>
    </div>
""")

st.write("")

if st.session_state.get("plan_email_sent"):
    st.success(f"📧 Sent to {st.session_state.email}")
else:
    if st.button("📧 Send this plan to my email", use_container_width=False):
        with st.spinner("Emailing your plan..."):
            sent, err = send_plan_email(
                st.session_state.email, st.session_state.username, plan_text
            )
        if sent:
            st.session_state["plan_email_sent"] = True
            st.success(f"📧 Sent your plan to {st.session_state.email}")
        else:
            st.error(f"⚠️ Couldn't email the plan: {err}")