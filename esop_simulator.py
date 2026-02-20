import math
import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime
import plotly.express as px
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from google import genai
from supabase import create_client, Client

# ==========================================
# NODE -1: SUPABASE AUTHENTICATION & DB
# ==========================================
st.set_page_config(page_title="Smart ESOP Advisory", page_icon="📈", layout="wide")

@st.cache_resource
def init_supabase() -> Client:
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception as e:
        st.error("⚠️ Supabase secrets not configured correctly in Streamlit Settings.")
        st.stop()

supabase = init_supabase()

# Initialize Auth State
if 'user' not in st.session_state:
    st.session_state.user = None
if 'portfolio_loaded' not in st.session_state:
    st.session_state.portfolio_loaded = False

# --- SECURE LOGIN PORTAL ---
if not st.session_state.user:
    st.title("🔐 Welcome to Smart ESOP Advisor")
    st.markdown("Please log in or create a secure account to track your ESOP portfolio and debt compounding.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        auth_mode = st.radio("Select Action", ["Login", "Sign Up"])
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        
        if st.button("Submit"):
            if auth_mode == "Sign Up":
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Account created successfully! You can now log in.")
                except Exception as e:
                    st.error(f"Sign Up Error: {str(e)}")
            else:
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = res.user
                    st.rerun()
                except Exception as e:
                    st.error("Invalid credentials. Please try again.")
    st.stop() # Halts execution of the main app until the user successfully authenticates

# --- FETCH SAVED PORTFOLIO ON LOGIN ---
if st.session_state.user and not st.session_state.portfolio_loaded:
    try:
        data = supabase.table("portfolios").select("*").eq("user_id", st.session_state.user.id).order("updated_at", desc=True).limit(1).execute()
        if data.data:
            row = data.data[0]
            st.session_state.def_ticker = row.get("ticker", "MEESHO.NS")
            st.session_state.def_emp = row.get("emp_status", "Active Employee")
            st.session_state.def_partner = row.get("partner", "Nuvama Wealth")
            st.session_state.def_opts = int(row.get("vested_options", 0))
            st.session_state.def_shares = int(row.get("total_shares", 0))
            st.session_state.def_fmv = float(row.get("fmv", 130.0))
            st.session_state.principal_loan = float(row.get("principal_loan", 0.0))
            st.session_state.def_date = row.get("sanction_date", str(datetime.date.today() - datetime.timedelta(days=1)))
        st.session_state.portfolio_loaded = True
    except Exception as e:
        st.session_state.portfolio_loaded = True # Proceed with defaults if fetch fails

# Provide fallbacks if no database entry was found
def_ticker = st.session_state.get("def_ticker", "MEESHO.NS")
def_emp = st.session_state.get("def_emp", "Active Employee")
def_partner = st.session_state.get("def_partner", "Nuvama Wealth")
def_opts = st.session_state.get("def_opts", 0)
def_shares = st.session_state.get("def_shares", 0)
def_fmv = st.session_state.get("def_fmv", 130.0)
def_date = st.session_state.get("def_date", str(datetime.date.today() - datetime.timedelta(days=1)))

# ==========================================
# NODE 0: FUNDING PARTNER TERMS DICTIONARY
# ==========================================
FUNDING_PARTNERS = {
    "Nuvama Wealth": {"doc_fee": 500, "processing_fee_pct": 0.0025, "margin_ltv": 0.50, "cure_period_days": 7, "interest_tiers": [(30, 0.075), (60, 0.085), (400, 0.0925)]},
    "Bajaj Financial Securities": {"doc_fee": 0, "processing_fee_pct": 0.0020, "margin_ltv": 0.50, "cure_period_days": 7, "interest_tiers": [(30, 0.0725), (90, 0.085), (400, 0.0925)]},
    "Infina Finance": {"doc_fee": 999, "processing_fee_pct": 0.0025, "margin_ltv": 0.50, "cure_period_days": 5, "interest_tiers": [(30, 0.0825), (90, 0.0875), (180, 0.0950), (400, 0.10)]},
    "360 ONE Prime": {"doc_fee": 0, "processing_fee_pct": 0.0015, "margin_ltv": 0.40, "cure_period_days": 1, "interest_tiers": [(90, 0.095), (180, 0.10), (400, 0.105)]}
}

# ==========================================
# NODE 1: THE DYNAMIC DEBT ENGINE
# ==========================================
def calculate_loan_debt(principal: float, days_elapsed: int, prepayments: list, terms: dict) -> float:
    if principal <= 0: return 0.0
    processing_fee_base = principal * terms["processing_fee_pct"]
    total_fees_at_closure = terms["doc_fee"] + processing_fee_base + (processing_fee_base * 0.18)
    total_interest, current_principal = 0.0, principal
    for day in range(1, days_elapsed + 1):
        daily_prepayment = sum(amt for p_day, amt in prepayments if p_day == day)
        current_principal = max(0, current_principal - daily_prepayment)
        daily_rate = 0.0
        for tier_day, rate in terms["interest_tiers"]:
            if day <= tier_day:
                daily_rate = rate / 365
                break
        total_interest += current_principal * daily_rate
    return round(current_principal + total_fees_at_closure + total_interest, 2)

# ==========================================
# NODE 1.5: ADVANCED TECHNICAL & MICRO DATA
# ==========================================
def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50.0
    delta = prices.diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    rs = up.ewm(com=period-1, adjust=False).mean() / down.ewm(com=period-1, adjust=False).mean()
    return round((100 - (100 / (1 + rs))).iloc[-1], 2)

@st.cache_data(ttl=60)
def get_market_data(ticker_symbol: str) -> dict:
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        current_price = info.get('currentPrice', stock.fast_info['last_price'])
        hist = stock.history(period="3mo")
        if not hist.empty:
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            volatility = min(hist['Close'].pct_change().dropna().std() * math.sqrt(252), 0.30)
            rsi_14 = calculate_rsi(hist['Close'], 14)
        else:
            sma_20, volatility, rsi_14 = current_price, 0.25, 50.0 
            
        has_analyst = 'targetMeanPrice' in info and info['targetMeanPrice'] is not None
        return {
            "current_price": round(current_price, 2), 
            "bull_target": round(info.get('targetHighPrice') if has_analyst else current_price * 1.25, 2),
            "base_target": round(info.get('targetMeanPrice') if has_analyst else current_price * 1.10, 2), 
            "bear_target": round(info.get('targetLowPrice') if has_analyst else current_price * 0.85, 2),
            "sma_20": round(sma_20, 2), "volatility": float(volatility), 
            "rsi_14": rsi_14, "has_analyst_data": has_analyst
        }
    except: return {"current_price": 0.0, "bull_target": 0.0, "base_target": 0.0, "bear_target": 0.0, "sma_20": 0.0, "volatility": 0.25, "rsi_14": 50.0, "has_analyst_data": False}

# ==========================================
# NODE 1.8: MULTI-SOURCE MACRO & RSS FETCHER
# ==========================================
def get_rss_news(url, max_items=4):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        root = ET.fromstring(urllib.request.urlopen(req, timeout=3).read())
        return [item.find('title').text for item in root.findall('.//item')[:max_items] if item.find('title') is not None]
    except: return []

@st.cache_data(ttl=1800)
def get_macro_context(ticker_symbol: str) -> dict:
    macro_data = {"nifty_price": 0.0, "usd_inr": 0.0, "india_vix": 0.0, "mc_news": [], "ticker_news": []}
    try: macro_data["nifty_price"] = round(yf.Ticker('^NSEI').fast_info['last_price'], 2)
    except: pass
    try: macro_data["usd_inr"] = round(yf.Ticker('INR=X').fast_info['last_price'], 2)
    except: pass
    try: macro_data["india_vix"] = round(yf.Ticker('^INDIAVIX').fast_info['last_price'], 2)
    except: pass
    macro_data["mc_news"] = get_rss_news("https://www.moneycontrol.com/rss/MCtopnews.xml", 4)
    query = urllib.parse.quote(f"{ticker_symbol.replace('.NS', '').replace('.BO', '')} stock India")
    macro_data["ticker_news"] = get_rss_news(f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en", 4)
    return macro_data

# ==========================================
# NODE 5: THE CIO MARKET SYNTHESIZER
# ==========================================
@st.cache_data(ttl=3600) 
def synthesize_market_intelligence(ticker: str, market_data: dict, macro_data: dict) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key: return "⚠️ Setup required: Gemini API Key not found."
    prompt = f"""
    You are a CIO analyzing {ticker} for an employee holding leveraged ESOPs.
    RAW MICRO: Price: ₹{market_data['current_price']}, 20-SMA: ₹{market_data['sma_20']}, RSI: {market_data.get('rsi_14', 50)} (>70 Overbought, <30 Oversold). News: {macro_data['ticker_news']}
    RAW MACRO: NIFTY: {macro_data['nifty_price']}, VIX: {macro_data['india_vix']} (>20 High Fear). News: {macro_data['mc_news']}
    Synthesize into a 3-bullet "Market Weather Report":
    * **Macro Trend:** [Assess NIFTY, VIX, Broad News]
    * **Micro Sentiment:** [Assess RSI vs SMA, Ticker News]
    * **Leverage Risk Level:** [Low/Medium/High - Safe to hold debt, or accelerate de-risking?]
    Keep it brief, professional, no fluff.
    """
    try: return genai.Client(api_key=api_key).models.generate_content(model='gemini-2.5-pro', contents=prompt).text
    except Exception as e: return f"⚠️ Engine Offline: {str(e)}"

# ==========================================
# NODE 2: THE LIQUIDATION & TAX ENGINE
# ==========================================
def calculate_liquidation_strategy(daily_debt: float, share_price: float, total_shares: int) -> dict:
    if share_price <= 0 or daily_debt <= 0: return {}
    net_realization = share_price * (1 - 0.0014) # Total margin
