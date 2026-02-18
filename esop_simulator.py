import math
import os
import yfinance as yf
import streamlit as st
import PyPDF2
from google import genai
from google.genai import types

# ==========================================
# NODE 1: THE DEBT ENGINE (State)
# ==========================================
def calculate_nuvama_debt(principal: float, days_elapsed: int, prepayment_amt: float = 0, prepayment_day: int = 0) -> float:
    doc_fee = 500
    processing_fee_base = principal * 0.0025 
    processing_fee_gst = processing_fee_base * 0.18 
    total_fees_at_closure = doc_fee + processing_fee_base + processing_fee_gst
    
    total_interest = 0.0
    current_principal = principal
    
    for day in range(1, days_elapsed + 1):
        if prepayment_amt > 0 and day > prepayment_day:
            current_principal = max(0, principal - prepayment_amt)
            
        if day <= 30:
            daily_rate = 0.075 / 365
        elif day <= 60:
            daily_rate = 0.085 / 365
        else:
            daily_rate = 0.0925 / 365
            
        total_interest += current_principal * daily_rate
        
    final_principal = principal if days_elapsed <= prepayment_day else max(0, principal - prepayment_amt)
    return round(final_principal + total_fees_at_closure + total_interest, 2)

# ==========================================
# NODE 1.5: THE MARKET FEED (Live Data)
# ==========================================
@st.cache_data(ttl=60)
def get_live_stock_price(ticker_symbol: str) -> float:
    try:
        stock = yf.Ticker(ticker_symbol)
        return round(stock.fast_info['last_price'], 2)
    except:
        return 0.0

# ==========================================
# NODE 2: THE LIQUIDATION ENGINE (Stream)
# ==========================================
def calculate_liquidation_strategy(daily_debt: float, share_price: float, total_shares: int) -> dict:
    if share_price <= 0: return {}

    nuvama_unpledge_fee = 0.0002 
    nuvama_brokerage = 0.0002    
    stt_tax = 0.001              
    total_sell_margin = nuvama_unpledge_fee + nuvama_brokerage + stt_tax
    
    net_realization_per_share = share_price * (1 - total_sell_margin)
    shares_to_sell = math.ceil(daily_debt / net_realization_per_share)
    remaining_shares = total_shares - shares_to_sell
    unlocked_wealth = remaining_shares * share_price
    
    return {
        "net_pocketed": net_realization_per_share,
        "shares_to_sell": shares_to_sell,
        "remaining_shares": remaining_shares,
        "unlocked_wealth": unlocked_wealth
    }

# ==========================================
# NODE 2.5: THE DETERMINISTIC TAX ENGINE
# ==========================================
def calculate_taxes(shares_sold: int, sell_price: float, fmv_on_exercise: float, holding_days: int) -> dict:
    """Calculates exact Indian Capital Gains Tax (STCG 20% / LTCG 12.5%)."""
    # Acquisition cost for ESOPs is the Fair Market Value (FMV) on the date of exercise
    capital_gains = (sell_price - fmv_on_exercise) * shares_sold
    
    if capital_gains <= 0:
        return {"tax_type": "No Gains / Capital Loss", "tax_liability": 0.0}
        
    if holding_days <= 365:
        tax_amount = capital_gains * 0.20 # 20% STCG
        return {"tax_type": "STCG (20%)", "tax_liability": round(tax_amount, 2)}
    else:
        # 12.5% LTCG (Standard ₹1.25L exemption applied)
        taxable_gains = max(0, capital_gains - 125000)
        tax_amount = taxable_gains * 0.125
        return {"tax_type": "LTCG (12.5%)", "tax_liability": round(tax_amount, 2)}

# ==========================================
# NODE 4: THE RAG AI STRATEGIST
# ==========================================
def generate_ai_insights(debt: float, share_price: float, shares_to_sell: int, remaining: int, margin_call: float, tax_data: dict, document_text: str):
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        return "⚠️ Error: Gemini API Key not found."
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an elite quantitative wealth advisor. 
    
    Here is the exact deterministic data for the portfolio today:
    - Current Loan Debt: ₹{debt:,.2f}
    - Live Share Price: ₹{share_price:,.2f}
    - Shares required to sell today to break even: {shares_to_sell:,}
    - Shares kept free and clear: {remaining:,}
    - Margin Call 'Doomsday' Price: ₹{margin_call:,.2f}
    
    TAX LIABILITY IF SOLD TODAY:
    - Tax Bracket Triggered: {tax_data['tax_type']}
    - Exact Tax Owed: ₹{tax_data['tax_liability']:,.2f}

    OFFICIAL UPLOADED DOCUMENTS (Read carefully for rules, penalties, and terms):
    {document_text if document_text else "No documents provided by the user."}

    Based STRICTLY on the deterministic math provided above and the rules within the uploaded documents, output a highly specific, 3-bullet arbitrage strategy. Do not guess tax rates or loan terms; rely entirely on the numbers and text provided here.
    """
    
    try:
        response = client.models.generate_content(model='gemini-2.5-pro', contents=prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

# ==========================================
# NODE 3: THE WEB DASHBOARD (Streamlit)
# ==========================================
st.set_page_config(page_title="ESOP Arbitrage Simulator", page_icon="📈", layout="wide")
st.title("📈 ESOP Liquidation & Arbitrage Simulator")

# --- SIDEBAR: User Inputs ---
st.sidebar.header("⚙️ Your ESOP Details")
total_shares = st.sidebar.number_input("Total Vested Shares", min_value=1, value=51000, step=1000)
principal_loan = st.sidebar.number_input("Total Loan Principal (₹)", min_value=100000, value=3000000, step=100000)
days_held = st.sidebar.slider("Days Loan Held", min_value=1, max_value=400, value=45)
ticker = st.sidebar.text_input("Live Ticker Symbol", value="MEESHO.NS")

st.sidebar.divider()
st.sidebar.subheader("💸 Short-Term Liquidity")
prepayment_amt = st.sidebar.number_input("Cash Injection Amount (₹)", min_value=0, value=0, step=100000)
prepayment_day = st.sidebar.slider("Payment Made on Day", min_value=1, max_value=365, value=15)

st.sidebar.divider()
st.sidebar.subheader("⚖️ Tax Variables")
fmv_on_exercise = st.sidebar.number_input("FMV on Exercise Date (₹)", min_value=1.0, value=80.0, step=1.0, help="The fair market value of the share on the exact day you exercised. Used as the base for Capital Gains.")

st.sidebar.divider()
st.sidebar.subheader("📄 Rulebook Upload")
uploaded_file = st.sidebar.file_uploader("Upload Term Sheets / ESOP Policy (PDF)", type="pdf")
document_text = ""
if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        document_text += page.extract_text() + "\n"
    st.sidebar.success("Document ingested successfully!")

# --- MAIN DASHBOARD ---
todays_debt = calculate_nuvama_debt(principal_loan, days_held, prepayment_amt, prepayment_day)
live_price = get_live_stock_price(ticker)

if live_price > 0:
    strategy = calculate_liquidation_strategy(todays_debt, live_price, total_shares)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Share Price", f"₹{live_price:,.2f}")
    col2.metric(f"Debt (Day {days_held})", f"₹{todays_debt:,.2f}")
    col3.metric("Net Pocketed / Share", f"₹{strategy['net_pocketed']:,.2f}")
    
    st.divider()
    
    st.subheader("Execution Plan & Tax Impact")
    col4, col5 = st.columns(2)
    col4.metric("Shares to Sell Today", f"{strategy['shares_to_sell']:,}", "To Break Even on Loan", delta_color="inverse")
    col5.metric("Remaining Wealth", f"₹{strategy['unlocked_wealth']:,.2f}", f"{strategy['remaining_shares']:,} Shares Free & Clear")
    
    # Calculate Deterministic Taxes
    tax_data = calculate_taxes(strategy['shares_to_sell'], live_price, fmv_on_exercise, days_held)
    st.warning(f"🏛️ **Estimated Tax Liability:** You are triggering **{tax_data['tax_type']}**. The exact tax owed on this sale is **₹{tax_data['tax_liability']:,.2f}**.")
    
    st.divider()
    
    st.subheader("🛡️ Risk Analysis")
    total_sell_margin = 0.0002 + 0.0002 + 0.001
    danger_price = todays_debt / (total_shares * (1 - total_sell_margin))
    
    if live_price <= danger_price:
        st.error(f"🚨 WARNING: UNDERWATER. At ₹{live_price}, your shares cannot cover the loan.")
    else:
        st.info(f"📉 **Margin Call Price: ₹{danger_price:,.2f}**")
        
    # --- NODE 4 UI: AI ADVISOR BUTTON ---
    st.divider()
    st.subheader("🧠 Document-Grounded AI Advisor")
    if st.button("Generate Strategy"):
        with st.spinner("Analyzing rules, taxes, and debt..."):
            insights = generate_ai_insights(todays_debt, live_price, strategy['shares_to_sell'], strategy['remaining_shares'], danger_price, tax_data, document_text)
            st.success(insights)
else:
    st.error("Could not fetch live market data. Please check the ticker symbol.")
