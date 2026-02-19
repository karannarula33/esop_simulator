import math
import os
import pandas as pd
import yfinance as yf
import streamlit as st
import PyPDF2
from google import genai
from google.genai import types

# ==========================================
# NODE 1: THE DEBT ENGINE
# ==========================================
def calculate_nuvama_debt(principal: float, days_elapsed: int, prepayments: list) -> float:
    if principal <= 0: return 0.0
    doc_fee = 500
    processing_fee_base = principal * 0.0025 
    processing_fee_gst = processing_fee_base * 0.18 
    total_fees_at_closure = doc_fee + processing_fee_base + processing_fee_gst
    
    total_interest = 0.0
    current_principal = principal
    
    for day in range(1, days_elapsed + 1):
        daily_prepayment = sum(amt for p_day, amt in prepayments if p_day == day)
        current_principal = max(0, current_principal - daily_prepayment)
            
        if day <= 30: daily_rate = 0.075 / 365
        elif day <= 60: daily_rate = 0.085 / 365
        else: daily_rate = 0.0925 / 365
            
        total_interest += current_principal * daily_rate
        
    return round(current_principal + total_fees_at_closure + total_interest, 2)

# ==========================================
# NODE 1.5: THE MARKET & NEWS FEED
# ==========================================
@st.cache_data(ttl=60)
def get_live_stock_price(ticker_symbol: str) -> float:
    try:
        return round(yf.Ticker(ticker_symbol).fast_info['last_price'], 2)
    except:
        return 0.0

@st.cache_data(ttl=3600)
def get_market_intelligence(ticker_symbol: str, api_key: str) -> str:
    """Fetches financial data & news, then uses AI to generate a concise macro summary."""
    if not api_key: return "⚠️ API Key required for Market Intelligence."
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        news = stock.news[:4]
        
        financial_context = f"""
        Ticker: {ticker_symbol}
        Current Price: {info.get('currentPrice', 'N/A')}
        52 Week High/Low: {info.get('fiftyTwoWeekHigh', 'N/A')} / {info.get('fiftyTwoWeekLow', 'N/A')}
        Market Cap: {info.get('marketCap', 'N/A')}
        Trailing P/E: {info.get('trailingPE', 'N/A')}
        Recent News Headlines: {[n['title'] for n in news]}
        """
        
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Analyze this raw financial data for {ticker_symbol}: {financial_context}
        Provide a highly concise, 3-bullet summary covering:
        1. The company's current market position/trajectory.
        2. Broader industry/country macroeconomic landscape sentiment.
        3. A quick summary of recent news catalysts.
        Keep it brief, articulate, and strictly factual. Maximum 100 words total.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Unable to fetch market intelligence: {str(e)}"

# ==========================================
# NODE 2: THE LIQUIDATION & TAX ENGINE
# ==========================================
def calculate_liquidation_strategy(daily_debt: float, share_price: float, total_shares: int) -> dict:
    if share_price <= 0 or daily_debt <= 0: return {}
    total_sell_margin = 0.0002 + 0.0002 + 0.001 
    net_realization_per_share = share_price * (1 - total_sell_margin)
    shares_to_sell = math.ceil(daily_debt / net_realization_per_share)
    return {
        "net_pocketed": net_realization_per_share,
        "shares_to_sell": shares_to_sell,
        "remaining_shares": total_shares - shares_to_sell,
        "unlocked_wealth": (total_shares - shares_to_sell) * share_price
    }

def calculate_taxes(shares_sold: int, sell_price: float, fmv_on_exercise: float, holding_days: int) -> dict:
    capital_gains = (sell_price - fmv_on_exercise) * shares_sold
    if capital_gains <= 0: return {"tax_type": "No Gains / Capital Loss", "tax_liability": 0.0}
    if holding_days <= 365: return {"tax_type": "STCG (20%)", "tax_liability": round(capital_gains * 0.20, 2)}
    return {"tax_type": "LTCG (12.5%)", "tax_liability": round(max(0, capital_gains - 125000) * 0.125, 2)}

# ==========================================
# NODE 4: THE TRAINED QUANTITATIVE AGENT
# ==========================================
def generate_ai_insights(user_query: str, emp_status: str, total_shares: int, debt: float, share_price: float, strategy: dict, margin_call: float, tax_data: dict, user_goals: str):
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key: return "⚠️ Error: Gemini API Key not found."
    
    trading_constraint = "SUBJECT TO STRICT 20-DAY QUARTERLY TRADING WINDOWS." if emp_status == "Active Employee" else "NO TRADING WINDOW RESTRICTIONS (Ex-Employee)."
    
    prompt = f"""
    You are an elite quantitative wealth advisor. 
    
    --- USER CONTEXT ---
    Status: {emp_status}
    Trading Rules: {trading_constraint}
    Query: "{user_query}"
    Goals: {user_goals if user_goals else 'Clear debt efficiently.'}
    
    --- FINANCIAL STATE ---
    Total Shares: {total_shares:,} | Loan Debt: ₹{debt:,.2f} | Share Price: ₹{share_price:,.2f}
    Min Shares to Clear Debt: {strategy.get('shares_to_sell', 0):,} | Margin Call: ₹{margin_call:,.2f}
    Tax Trigger: {tax_data['tax_type']} (Liability: ₹{tax_data['tax_liability']:,.2f})
    
    --- EXECUTION FRAMEWORK ---
    1. Answer the user's query directly and mathematically.
    2. If suggesting a selling strategy, provide explicit tranche sizes (e.g., "Sell X shares today").
    3. CRITICAL: Be extremely concise and articulate. Use short bullet points. Cut all fluff, introductory greetings, and verbose explanations. Provide pure, actionable financial strategy.
    """
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-pro', contents=prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

# ==========================================
# NODE 3: THE WEB DASHBOARD (Streamlit UI)
# ==========================================
st.set_page_config(page_title="Smart ESOP Advisory", page_icon="📈", layout="wide")
st.title("📈 Smart ESOP Investment & Advisory Platform")

# --- SIDEBAR: Portfolio Configuration ---
st.sidebar.header("⚙️ Portfolio Configuration")
ticker = st.sidebar.text_input("Live Ticker Symbol", value="MEESHO.NS")
emp_status = st.sidebar.radio("Employment Status", ["Active Employee", "Ex-Employee"])

st.sidebar.divider()
st.sidebar.subheader("🧮 Loan & Equity Calculator")
grant_type = st.sidebar.selectbox("Grant Type", ["Meesho Ltd (1:49)", "Meesho Inc (1:60)"])
entitlement_ratio = 49 if "Ltd" in grant_type else 60

vested_esops = st.sidebar.number_input("Vested ESOPs (Options)", min_value=0, value=0, step=100)
total_shares = vested_esops * entitlement_ratio
st.sidebar.caption(f"Calculated Total Shares: **{total_shares:,}**")

fmv_on_exercise = st.sidebar.number_input("FMV on Exercise Date (₹)", min_value=0.0, value=130.0, step=1.0)

# Session State for Loan Amount
if 'principal_loan' not in st.session_state:
    st.session_state.principal_loan = 0.0

if st.sidebar.button("Calculate Exact Loan Needed"):
    if vested_esops > 0 and fmv_on_exercise > 0:
        exercise_payable = vested_esops * 1 # INR 1 per option
        perq_value = (total_shares * fmv_on_exercise) - exercise_payable
        tax_rate = 0.30 if emp_status == "Active Employee" else 0.4274
        perq_tax = perq_value * tax_rate
        st.session_state.principal_loan = round(exercise_payable + perq_tax, 2)
    else:
        st.sidebar.warning("Enter ESOPs and FMV first.")

principal_loan = st.sidebar.number_input("Total Loan Principal (₹)", min_value=0.0, value=float(st.session_state.principal_loan), step=10000.0)
days_held = st.sidebar.slider("Days Loan Held", min_value=1, max_value=400, value=1)

st.sidebar.divider()
st.sidebar.subheader("💸 Cash Injections (Multiple)")
if 'prepayments_df' not in st.session_state:
    st.session_state.prepayments_df = pd.DataFrame({"Day": [15], "Amount (₹)": [0]})
edited_prepayments = st.sidebar.data_editor(st.session_state.prepayments_df, num_rows="dynamic", hide_index=True)
prepayments_list = list(zip(edited_prepayments["Day"], edited_prepayments["Amount (₹)"]))

# --- MAIN DASHBOARD ---
if total_shares == 0 or principal_loan == 0:
    st.info("👋 Welcome. Please enter your Vested ESOPs and calculate your Loan Principal in the sidebar to begin.")
else:
    todays_debt = calculate_nuvama_debt(principal_loan, days_held, prepayments_list)
    live_price = get_live_stock_price(ticker)

    if live_price > 0:
        strategy = calculate_liquidation_strategy(todays_debt, live_price, total_shares)
        
        # 1. TOP SECTION: Default Execution Strategy
        st.header("Today's Default Execution Strategy")
        col1, col2, col3 = st.columns(3)
        col1.metric("Live Share Price", f"₹{live_price:,.2f}")
        col2.metric("Outstanding Debt", f"₹{todays_debt:,.2f}")
        col3.metric("Net Pocketed / Share", f"₹{strategy['net_pocketed']:,.2f}")
        
        st.divider()
        col4, col5 = st.columns(2)
        col4.metric("Shares to Sell Today", f"{strategy['shares_to_sell']:,}", "To Break Even on Loan", delta_color="inverse")
        col5.metric("Remaining Shares (Free & Clear)", f"{strategy['remaining_shares']:,}", f"Value: ₹{strategy['unlocked_wealth']:,.2f}", delta_color="normal")
        
        # 2. MIDDLE SECTION: Market Intelligence
        st.divider()
        st.subheader("🌐 Market & Financial Intelligence")
        with st.spinner("Compiling macroeconomic and financial data..."):
            api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
            market_brief = get_market_intelligence(ticker, api_key)
            st.info(market_brief)

        # 3. BOTTOM SECTION: Goal Setting & Strategy Chat
        st.divider()
        st.subheader("🎯 Custom Strategy & AI Advisor")
        user_goals = st.text_input("Optional: Define specific goals (e.g., 'Need ₹20L in 3 months for a house')")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., 'Develop a minimal-tax selling strategy.'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Computing concise execution strategy..."):
                    total_sell_margin = 0.0002 + 0.0002 + 0.001
                    danger_price = todays_debt / (total_shares * (1 - total_sell_margin))
                    tax_data = calculate_taxes(strategy['shares_to_sell'], live_price, fmv_on_exercise, days_held)
                    
                    response = generate_ai_insights(prompt, emp_status, total_shares, todays_debt, live_price, strategy, danger_price, tax_data, user_goals)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Could not fetch live market data. Please check the ticker symbol.")
