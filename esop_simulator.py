import math
import yfinance as yf
import streamlit as st

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
        # Apply the cash injection to reduce the principal on the exact day it happens
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
@st.cache_data(ttl=60) # Caches the price for 60 seconds so it doesn't spam the NSE
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
# NODE 3: THE WEB DASHBOARD (Streamlit)
# ==========================================
st.set_page_config(page_title="ESOP Arbitrage Simulator", page_icon="📈", layout="wide")

st.title("📈 ESOP Liquidation & Arbitrage Simulator")
st.markdown("Calculate your exact exit strategy to clear Nuvama funding debt based on **live NSE market data**.")

# --- SIDEBAR: User Inputs ---
st.sidebar.header("⚙️ Your ESOP Details")
st.sidebar.markdown("Enter your specific variables below:")

total_shares = st.sidebar.number_input("Total Vested Shares", min_value=1, value=51000, step=1000)
# We use 30 Lakhs as the default, but you can change it here!
principal_loan = st.sidebar.number_input("Total Loan Principal (₹)", min_value=100000, value=3000000, step=100000)
days_held = st.sidebar.slider("Days Loan Held", min_value=1, max_value=365, value=45)
ticker = st.sidebar.text_input("Live Ticker Symbol", value="MEESHO.NS")
st.sidebar.divider()
st.sidebar.subheader("💸 Short-Term Liquidity")
st.sidebar.caption("Simulate making an early cash payment to reduce the principal.")
prepayment_amt = st.sidebar.number_input("Cash Injection Amount (₹)", min_value=0, value=0, step=100000)
prepayment_day = st.sidebar.slider("Payment Made on Day", min_value=1, max_value=365, value=15)

# --- MAIN DASHBOARD ---
st.header("Today's Strategy")

# Run the Math
todays_debt = calculate_nuvama_debt(principal_loan, days_held, prepayment_amt, prepayment_day)
live_price = get_live_stock_price(ticker)

if live_price > 0:
    strategy = calculate_liquidation_strategy(todays_debt, live_price, total_shares)
    
    # Top Row Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Share Price", f"₹{live_price:,.2f}")
    col2.metric(f"Debt (Day {days_held})", f"₹{todays_debt:,.2f}")
    col3.metric("Net Pocketed / Share", f"₹{strategy['net_pocketed']:,.2f}")
    
    st.divider()
    
    # Bottom Row Action Plan
    st.subheader("Execution Plan")
    col4, col5 = st.columns(2)
    
    col4.metric(
        label="Shares to Sell Today", 
        value=f"{strategy['shares_to_sell']:,}", 
        delta="To Break Even on Loan", 
        delta_color="inverse"
    )
    
    col5.metric(
        label="Remaining Shares (Free & Clear)", 
        value=f"{strategy['remaining_shares']:,}", 
        delta=f"Value: ₹{strategy['unlocked_wealth']:,.2f}",
        delta_color="normal"
    )
    
    st.divider()
    
    # --- DEFENSIVE METRIC: STOP LOSS ---
    st.subheader("🛡️ Risk Analysis")
    
    # Calculate the exact price where selling ALL shares perfectly equals the debt
    # Formula: Debt / (Total Shares * Net Margin)
    total_sell_margin = 0.0002 + 0.0002 + 0.001
    danger_price = todays_debt / (total_shares * (1 - total_sell_margin))
    
    if live_price <= danger_price:
        st.error(f"🚨 WARNING: UNDERWATER. At ₹{live_price}, your total shares cannot cover the loan.")
    else:
        buffer_percent = ((live_price - danger_price) / live_price) * 100
        st.warning(f"📉 **Margin Call Price: ₹{danger_price:,.2f}**")
        st.caption(f"If Meesho shares drop below ₹{danger_price:,.2f}, your entire ESOP pool will be wiped out by the loan debt. You currently have a {buffer_percent:.1f}% safety buffer.")

else:
    st.error("Could not fetch live market data. Please check the ticker symbol.")