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
if 'user' not in st.session_state: st.session_state.user = None
if 'portfolio_loaded' not in st.session_state: st.session_state.portfolio_loaded = False

# Database Safe-Casting Functions
def safe_int(val, fallback=0):
    try: return int(float(val)) if val is not None else fallback
    except: return fallback

def safe_float(val, fallback=0.0):
    try: return float(val) if val is not None else fallback
    except: return fallback

# --- SECURE LOGIN PORTAL (ANTI-GHOSTING WRAPPER) ---
if not st.session_state.user:
    login_container = st.empty()
    with login_container.container():
        st.title("🔐 Welcome to Smart ESOP Advisor")
        st.markdown("Please log in or create a secure account to track your ESOP portfolio and debt compounding.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            auth_mode = st.radio("Select Action", ["Login", "Sign Up"], key="auth_mode_radio")
            email = st.text_input("Email Address", key="login_email_input")
            password = st.text_input("Password", type="password", key="login_pass_input")
            
            if st.button("Submit", key="auth_submit_btn"):
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
                        login_container.empty() # EXPLICITLY DESTROY THE UI TO PREVENT GHOSTING
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login Error: {str(e)}")
    st.stop() 

# --- FETCH SAVED PORTFOLIO ON LOGIN ---
if st.session_state.user and not st.session_state.portfolio_loaded:
    try:
        data = supabase.table("portfolios").select("*").eq("user_id", st.session_state.user.id).order("updated_at", desc=True).limit(1).execute()
        if data.data:
            row = data.data[0]
            st.session_state.def_ticker = str(row.get("ticker", "MEESHO.NS"))
            st.session_state.def_emp = str(row.get("emp_status", "Active Employee"))
            st.session_state.def_partner = str(row.get("partner", "Nuvama Wealth"))
            st.session_state.def_opts = safe_int(row.get("vested_options"))
            st.session_state.def_shares = safe_int(row.get("total_shares"))
            st.session_state.def_fmv = safe_float(row.get("fmv", 130.0))
            st.session_state.principal_loan = safe_float(row.get("principal_loan"))
            
            dt_str = row.get("sanction_date")
            st.session_state.def_date = str(dt_str) if dt_str else str(datetime.date.today() - datetime.timedelta(days=1))
        st.session_state.portfolio_loaded = True
    except Exception as e:
        st.session_state.portfolio_loaded = True 

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
    "Nuvama Wealth": {"doc_fee": 500, "processing_fee_pct": 0.0025, "margin_ltv": 0.50, "cure_period_days": 7, "interest_tiers": [(30, 0.07), (60, 0.08), (400, 0.09)]},
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

@st.cache_data(ttl=300) 
def get_market_data(ticker_symbol: str) -> dict:
    try:
        stock = yf.Ticker(ticker_symbol)
        current_price = float(stock.fast_info['last_price'])
        
        hist = stock.history(period="3mo")
        if not hist.empty:
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            volatility = min(hist['Close'].pct_change().dropna().std() * math.sqrt(252), 0.30)
            rsi_14 = calculate_rsi(hist['Close'], 14)
        else:
            sma_20, volatility, rsi_14 = current_price, 0.25, 50.0 
            
        try:
            info = stock.info
            has_analyst = 'targetMeanPrice' in info and info['targetMeanPrice'] is not None
            bull = float(info.get('targetHighPrice', current_price * 1.25)) if has_analyst else current_price * 1.25
            base = float(info.get('targetMeanPrice', current_price * 1.10)) if has_analyst else current_price * 1.10
            bear = float(info.get('targetLowPrice', current_price * 0.85)) if has_analyst else current_price * 0.85
        except:
            has_analyst = False
            bull, base, bear = current_price * 1.25, current_price * 1.10, current_price * 0.85
        
        return {
            "current_price": round(current_price, 2), "bull_target": round(bull, 2),
            "base_target": round(base, 2), "bear_target": round(bear, 2),
            "sma_20": round(sma_20, 2), "volatility": float(volatility), 
            "rsi_14": rsi_14, "has_analyst_data": has_analyst
        }
    except Exception as e: 
        return {"current_price": 0.0, "error": str(e)}

# ==========================================
# NODE 1.8: MULTI-SOURCE MACRO & RSS FETCHER
# ==========================================
def get_rss_news(url, max_items=4):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        root = ET.fromstring(urllib.request.urlopen(req, timeout=3).read())
        return [item.find('title').text for item in root.findall('.//item')[:max_items] if item.find('title') is not None]
    except: return []

@st.cache_data(ttl=900) 
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
@st.cache_data(ttl=1800) 
def synthesize_market_intelligence(ticker: str, market_data: dict, macro_data: dict) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key: return "⚠️ Setup required: Gemini API Key not found."
    
    prompt = f"""
    You are a CIO analyzing {ticker} for an employee holding leveraged ESOPs.
    RAW MICRO: Price: ₹{market_data.get('current_price', 0)}, 20-SMA: ₹{market_data.get('sma_20', 0)}, RSI: {market_data.get('rsi_14', 50)} (>70 Overbought, <30 Oversold). News: {macro_data.get('ticker_news', [])}
    RAW MACRO: NIFTY: {macro_data.get('nifty_price', 0)}, VIX: {macro_data.get('india_vix', 0)} (>20 High Fear). News: {macro_data.get('mc_news', [])}
    Synthesize into a 3-bullet "Market Weather Report":
    * **Macro Trend:** [Assess NIFTY, VIX, Broad News]
    * **Micro Sentiment:** [Assess RSI vs SMA, Ticker News]
    * **Leverage Risk Level:** [Low/Medium/High - Safe to hold debt, or accelerate de-risking?]
    Keep it brief, professional, no fluff.
    """
    try: 
        with genai.Client(api_key=api_key) as client:
            return client.models.generate_content(model='gemini-2.5-pro', contents=prompt).text
    except Exception as e: return f"⚠️ Engine Offline: {str(e)}"

# ==========================================
# NODE 2: THE LIQUIDATION & TAX ENGINE
# ==========================================
def calculate_liquidation_strategy(daily_debt: float, share_price: float, total_shares: int) -> dict:
    if share_price <= 0 or daily_debt <= 0: return {}
    net_realization = share_price * (1 - 0.0014) # Total margin
    shares_to_sell = math.ceil(daily_debt / net_realization)
    return {"net_pocketed": net_realization, "shares_to_sell": shares_to_sell, "remaining_shares": total_shares - shares_to_sell, "unlocked_wealth": (total_shares - shares_to_sell) * share_price}

def calculate_taxes(shares_sold: int, sell_price: float, fmv_on_exercise: float, holding_days: int) -> dict:
    capital_gains = (sell_price - fmv_on_exercise) * shares_sold
    if capital_gains <= 0: return {"tax_type": "No Gains / Capital Loss", "tax_liability": 0.0}
    if holding_days <= 365: return {"tax_type": "STCG (20%)", "tax_liability": round(capital_gains * 0.20, 2)}
    return {"tax_type": "LTCG (12.5%)", "tax_liability": round(max(0, capital_gains - 125000) * 0.125, 2)}

# ==========================================
# NODE 2.5: MEAN-REVERTING PROJECTION 
# ==========================================
def generate_projection_data(principal: float, total_shares: int, fmv_on_exercise: float, market_data: dict, prepayments: list, sanction_date: datetime.date, terms: dict):
    np.random.seed(42) 
    wealth_data, price_data, p0 = [], [], market_data['current_price']
    daily_drift = math.log(market_data['base_target'] / p0) / 365 if p0 > 0 else 0
    current_sim_price = p0
    
    for d in range(1, 400, 3):
        current_date = sanction_date + datetime.timedelta(days=d)
        debt = calculate_loan_debt(principal, d, prepayments, terms)
        
        if d > 1:
            step_vol = market_data['volatility'] * math.sqrt(3/252)
            z = np.random.normal(0, 1)
            linear_exp = p0 + ((market_data['base_target'] - p0) / 365) * d
            current_sim_price = current_sim_price * math.exp(daily_drift * 3 - 0.5 * step_vol**2 + step_vol * z) + ((linear_exp - current_sim_price) * 0.08)
            
        gross_value = current_sim_price * total_shares
        margin_call_level = debt / terms["margin_ltv"]
        tax_hit = calculate_taxes(total_shares, current_sim_price, fmv_on_exercise, d)['tax_liability']
        
        wealth_data.append({"Date": current_date, "Gross Portfolio Value (₹)": gross_value, "Net Wealth (₹)": max(0, gross_value - debt - tax_hit), "Margin Call Threshold (₹)": margin_call_level, "Total Debt (₹)": debt, "Underlying Share Price": f"₹{current_sim_price:,.2f}"})
        price_data.append({"Date": current_date, "Simulated Price (₹)": current_sim_price, "Bull Target (₹)": market_data['bull_target'], "Base Target (₹)": market_data['base_target'], "Bear Target (₹)": market_data['bear_target']})
        
    return pd.DataFrame(wealth_data).set_index("Date"), pd.DataFrame(price_data).set_index("Date")

# ==========================================
# NODE 4: THE CROSS-COMMUNICATING AGENT
# ==========================================
def generate_ai_insights(user_query: str, emp_status: str, total_shares: int, debt: float, market_data: dict, strategy: dict, margin_call: float, tax_data: dict, days_held: int, partner_name: str, terms: dict, market_intelligence: str, is_default: bool = False):
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key: return "⚠️ Error: Gemini API Key not found."
    
    constraint = "CRITICAL: The user is an Active Employee. ALL sales MUST be strictly scheduled during the '20-day quarterly trading windows'." if emp_status == "Active Employee" else "User is Ex-Employee. No blackout restrictions."
    
    state = f"""--- FINANCIAL STATE ---
    Partner: {partner_name} | LTV: {terms['margin_ltv']*100}% | Cure: {terms['cure_period_days']} days
    Shares: {total_shares:,} | Debt: ₹{debt:,.2f} | Days Held: {days_held}
    Price: ₹{market_data.get('current_price', 0):,.2f} | Tax Trigger: {tax_data.get('tax_type', 'Unknown')}
    --- MARKET WEATHER ---
    {market_intelligence}"""
    
    if is_default:
        prompt = f"""You are a quant advisor. Read state:\n{state}\nRules: {constraint}
        Output a concise 'Portfolio Health Check' (Margin risk, Tax state, and 1-sentence strategic takeaway from Market Weather). 
        End by explicitly asking the user to choose their primary intent. Format this EXACTLY as a vertical bulleted list so it is easy to read on mobile devices:
        
        * **[A] Aggressive Debt Elimination** (I want to clear my loan ASAP)
        * **[B] Maximize Long-Term Wealth** (I am willing to hold the loan for LTCG benefits and stock upside)
        * **[C] External Capital Extraction** (I need to liquidate a specific amount for personal goals)
        """
    else:
        prompt = f"You are an algorithmic execution engine. State:\n{state}\nRules: {constraint}\nUser replied: '{user_query}'. Generate a multi-tranche schedule.\nCRITICAL: Format as Phase 1, Phase 2, etc. Each phase MUST include:\n* Action: [Sell/Hold]\n* Timing: [Specific]\n* Target Price: [Rupee]\n* Macro Alignment: [Mandatory 1-sentence explaining how this tranche reacts specifically to the VIX, RSI, or Macro Weather report.]"
    
    try:
        with genai.Client(api_key=api_key) as client:
            return client.models.generate_content(model='gemini-2.5-pro', contents=prompt).text
    except Exception as e: return f"⚠️ AI Error: {str(e)}"

# ==========================================
# MAIN DASHBOARD UI
# ==========================================
st.title("📈 Smart ESOP Investment & Advisory Platform")

st.sidebar.header(f"👤 Logged in as: {st.session_state.user.email.split('@')[0]}")

if st.sidebar.button("🔄 Refresh Live Market Data", type="primary"):
    st.cache_data.clear() 
    st.rerun() 

if st.sidebar.button("🚪 Log Out"):
    st.session_state.user = None
    st.session_state.portfolio_loaded = False
    supabase.auth.sign_out()
    st.rerun()

st.sidebar.divider()
st.sidebar.header("⚙️ Portfolio Configuration")
ticker = st.sidebar.text_input("Live Ticker Symbol", value=def_ticker)
emp_status = st.sidebar.radio("Employment Status", ["Active Employee", "Ex-Employee"], index=0 if def_emp == "Active Employee" else 1)

st.sidebar.divider()
st.sidebar.subheader("🏦 Funding Partner Selection")
partner_index = list(FUNDING_PARTNERS.keys()).index(def_partner) if def_partner in FUNDING_PARTNERS else 0
selected_partner = st.sidebar.selectbox("Select your financer:", list(FUNDING_PARTNERS.keys()), index=partner_index)
active_terms = FUNDING_PARTNERS[selected_partner]

st.sidebar.divider()
st.sidebar.subheader("🧮 Loan & Equity Calculator")
vested_options = st.sidebar.number_input("Total Vested Options (ESOPs)", min_value=0, value=def_opts, step=100)
total_shares = st.sidebar.number_input("Total Resulting Shares", min_value=0, value=def_shares, step=100)
fmv_on_exercise = st.sidebar.number_input("FMV on Exercise Date (₹)", min_value=0.0, value=def_fmv, step=1.0)

if 'principal_loan' not in st.session_state: st.session_state.principal_loan = st.session_state.get("principal_loan", 0.0)
if 'calc_breakdown' not in st.session_state: st.session_state.calc_breakdown = ""

if st.sidebar.button("Calculate Exact Loan Needed"):
    if vested_options > 0 and total_shares > 0 and fmv_on_exercise > 0:
        exercise_payable = vested_options * 1
        perq_tax = ((total_shares * fmv_on_exercise) - exercise_payable) * (0.30 if emp_status == "Active Employee" else 0.4274)
        st.session_state.principal_loan = round(exercise_payable + perq_tax, 2)
        st.session_state.calc_breakdown = f"**Calculation Breakdown:**\n- Exercise Price: ₹{exercise_payable:,.2f}\n- Perquisite Tax: ₹{perq_tax:,.2f}\n- **Total Loan: ₹{st.session_state.principal_loan:,.2f}**"
    else: st.sidebar.warning("Enter Options, Shares, and FMV first.")

if st.session_state.calc_breakdown: st.sidebar.info(st.session_state.calc_breakdown)

principal_loan = st.sidebar.number_input("Total Loan Principal (₹)", min_value=0.0, value=safe_float(st.session_state.get("principal_loan", 0.0)), step=10000.0)

try:
    sanction_dt = datetime.datetime.strptime(def_date, "%Y-%m-%d").date() if def_date else datetime.date.today() - datetime.timedelta(days=1)
except:
    sanction_dt = datetime.date.today() - datetime.timedelta(days=1)

loan_sanction_date = st.sidebar.date_input("Loan Sanction Date", sanction_dt)
days_held = max(1, (datetime.date.today() - loan_sanction_date).days)
sim_days = st.sidebar.slider("Simulate Future Date (Days Held
