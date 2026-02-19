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
def get_company_news(ticker_symbol: str) -> list:
    """Pulls the latest news headlines for the ticker to ground the AI in reality."""
    try:
        news_data = yf.Ticker(ticker_symbol).news
        return [{"title": n["title"], "publisher": n["publisher"], "link": n["link"]} for n in news_data[:5]]
    except:
        return []

# ==========================================
# NODE 2: THE LIQUIDATION & TAX ENGINE
# ==========================================
def calculate_liquidation_strategy(daily_debt: float, share_price: float, total_shares: int) -> dict:
    if share_price <= 0 or daily_debt <= 0: return {}

    total_sell_margin = 0.0002 + 0.0002 + 0.001 # Unpledge + Brokerage + STT
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
def generate_ai_insights(user_query: str, debt: float, share_price: float, strategy: dict, margin_call: float, tax_data: dict, user_goals: str, live_news: list):
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key: return "⚠️ Error: Gemini API Key not found."
    
    # Format news for the AI
    news_context = "\n".join([f"- {n['title']} (Source: {n['publisher']})" for n in live_news]) if live_news else "No recent news available."
    
    prompt = f"""
    You are an elite, algorithmic quantitative wealth advisor. The user is asking: "{user_query}"
    
    --- 1. DETERMINISTIC FINANCIAL STATE ---
    - Current Loan Debt: ₹{debt:,.2f}
    - Live Share Price: ₹{share_price:,.2f}
    - Minimum Shares to Clear Debt Today: {strategy.get('shares_to_sell', 0):,}
    - Margin Call 'Doomsday' Price: ₹{margin_call:,.2f}
    - Current Tax Trigger: {tax_data['tax_type']} (Liability: ₹{tax_data['tax_liability']:,.2f})
    
    --- 2. USER'S DEFINED STRATEGIC GOALS ---
    {user_goals if user_goals else "User has not defined specific goals. Optimize for maximum wealth retention and minimum risk."}
    
    --- 3. LIVE MARKET SENTIMENT (REAL-TIME NEWS) ---
    {news_context}

    --- 4. INSTITUTIONAL KNOWLEDGE BASE (STRICT RULES) ---
    - NUVAMA LOAN TERMS: 100% exercise price funding. Interest: 7.5% p.a. (Days 1-30), 8.5% (Days 31-60), 9.25% (Days 61-365). Daily simple interest. Processing fee 0.25%+GST. Unpledge/Brokerage 0.02% each. No EMIs required.
    - MEESHO ESOP POLICY: [To be inserted once provided by user]

    --- EXECUTION FRAMEWORK (HOW TO THINK) ---
    Do not give vague advice. You MUST output a structured execution plan:
    1. Liquidity & Risk Assessment: How close are they to the margin call based on the news?
    2. Tax Optimization: Is it worth taking the Nuvama penalty interest to cross the 365-day LTCG threshold? 
    3. Tranche Sizing: Provide exact mathematical recommendations (e.g., "Sell X% today to de-risk, set a limit order for Y% at ₹[Target Price] to cover taxes, hold Z% for the user's stated goals").
    """
    
    try:
        client = genai.Client(api_key=api_key)
        # Notice we removed the Google Search tool. The AI is now strictly grounded in the exact news and data WE fed it above.
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
total_shares = st.sidebar.number_input("Total Vested Shares", min_value=0, value=0, step=1000)
principal_loan = st.sidebar.number_input("Total Loan Principal (₹)", min_value=0, value=0, step=100000)
days_held = st.sidebar.slider("Days Loan Held", min_value=1, max_value=400, value=1)
fmv_on_exercise = st.sidebar.number_input("FMV on Exercise Date (₹)", min_value=0.0, value=0.0, step=1.0)
ticker = st.sidebar.text_input("Live Ticker Symbol", value="MEESHO.NS")

st.sidebar.divider()
st.sidebar.subheader("💸 Cash Injections (Multiple)")
if 'prepayments_df' not in st.session_state:
    st.session_state.prepayments_df = pd.DataFrame({"Day": [15], "Amount (₹)": [0]})
edited_prepayments = st.sidebar.data_editor(st.session_state.prepayments_df, num_rows="dynamic", hide_index=True)
prepayments_list = list(zip(edited_prepayments["Day"], edited_prepayments["Amount (₹)"]))

# --- MAIN DASHBOARD ---
if total_shares == 0 or principal_loan == 0:
    st.info("👋 Welcome to your Quantitative Family Office. Please configure your baseline portfolio in the sidebar to generate the dashboard.")
else:
    todays_debt = calculate_nuvama_debt(principal_loan, days_held, prepayments_list)
    live_price = get_live_stock_price(ticker)
    live_news = get_company_news(ticker)

    if live_price > 0:
        strategy = calculate_liquidation_strategy(todays_debt, live_price, total_shares)
        
        # 1. Top Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Live Share Price", f"₹{live_price:,.2f}")
        col2.metric("Outstanding Debt", f"₹{todays_debt:,.2f}")
        col3.metric("Shares Req. to Clear Debt", f"{strategy['shares_to_sell']:,}")
        
        # 2. Strategic Goals Input
        st.divider()
        st.subheader("🎯 Define Your Strategic Goals")
        user_goals = st.text_area("What are you trying to achieve? (e.g., 'I need ₹50L liquid in 6 months for a new business, and want to hold the rest long-term'.)", height=100)
        
        # 3. Market Context & Data Transparency
        st.divider()
        with st.expander("📰 Live Market Context (Data fed to the AI)"):
            st.markdown("The AI is currently basing its tranche execution strategy on the following real-time data:")
            total_sell_margin = 0.0002 + 0.0002 + 0.001
            danger_price = todays_debt / (total_shares * (1 - total_sell_margin))
            tax_data = calculate_taxes(strategy['shares_to_sell'], live_price, fmv_on_exercise, days_held)
            
            st.markdown(f"- **Margin Call Risk:** Price ₹{danger_price:,.2f}")
            st.markdown(f"- **Current Tax Bracket:** {tax_data['tax_type']} (Liability: ₹{tax_data['tax_liability']:,.2f})")
            st.markdown("**Latest Market Headlines:**")
            if live_news:
                for item in live_news:
                    st.markdown(f"- [{item['title']}]({item['link']}) - *{item['publisher']}*")
            else:
                st.markdown("- *No breaking news found for this ticker today.*")

        # 4. AI Strategy Agent
        st.subheader("🧠 Investment Strategy Agent")
        st.caption("Ask your advisor to develop a selling plan based on your defined goals and current market conditions.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., 'Develop a 3-tranche selling strategy that fulfills my goals while minimizing tax liability.'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing goals, pricing tranches, and computing tax liabilities..."):
                    response = generate_ai_insights(prompt, todays_debt, live_price, strategy, danger_price, tax_data, user_goals, live_news)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Could not fetch live market data. Please check the ticker symbol.")
