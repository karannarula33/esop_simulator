import math
import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime
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
# NODE 1.5: ADVANCED MARKET & GRANULAR DATA
# ==========================================
@st.cache_data(ttl=60)
def get_market_data(ticker_symbol: str) -> dict:
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        current_price = info.get('currentPrice', stock.fast_info['last_price'])
        
        hist = stock.history(period="3mo")
        if not hist.empty:
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            daily_returns = hist['Close'].pct_change().dropna()
            volatility = daily_returns.std() * math.sqrt(252) 
        else:
            sma_20 = current_price
            volatility = 0.40 
            
        bull_target = info.get('targetHighPrice', current_price * 1.25)
        base_target = info.get('targetMeanPrice', current_price * 1.10)
        bear_target = info.get('targetLowPrice', current_price * 0.85)
        
        return {
            "current_price": round(current_price, 2),
            "bull_target": round(bull_target, 2),
            "base_target": round(base_target, 2),
            "bear_target": round(bear_target, 2),
            "sma_20": round(sma_20, 2),
            "volatility": float(volatility)
        }
    except:
        return {"current_price": 0.0, "bull_target": 0.0, "base_target": 0.0, "bear_target": 0.0, "sma_20": 0.0, "volatility": 0.40}

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
# NODE 2.5: GEOMETRIC BROWNIAN MOTION PROJECTION
# ==========================================
def generate_projection_data(principal: float, total_shares: int, fmv_on_exercise: float, market_data: dict, prepayments: list, sanction_date: datetime.date):
    np.random.seed(42) 
    days_to_sim = list(range(1, 400, 3)) 
    data = []
    
    p0 = market_data['current_price']
    vol = market_data['volatility']
    
    drift_base = math.log(market_data['base_target'] / p0) / 365
    drift_bull = math.log(market_data['bull_target'] / p0) / 365
    drift_bear = math.log(market_data['bear_target'] / p0) / 365
    
    price_base = p0
    price_bull = p0
    price_bear = p0
    
    for d in days_to_sim:
        current_date = sanction_date + datetime.timedelta(days=d)
        debt = calculate_nuvama_debt(principal, d, prepayments)
        
        z = np.random.normal(0, 1)
        step_vol = vol * math.sqrt(3/252)
        
        if d > 1:
            price_base *= math.exp(drift_base * 3 - 0.5 * step_vol**2 + step_vol * z)
            price_bull *= math.exp(drift_bull * 3 - 0.5 * step_vol**2 + step_vol * z)
            price_bear *= math.exp(drift_bear * 3 - 0.5 * step_vol**2 + step_vol * z)
            
        def get_true_net_wealth(sim_price):
            gross_val = sim_price * total_shares
            tax_hit = calculate_taxes(total_shares, sim_price, fmv_on_exercise, d)['tax_liability']
            return gross_val - debt - tax_hit

        # The exact net wealth line where Margin Call triggers (50% LTV)
        # Margin call triggers when Gross Value = 2 * Debt. 
        # Net wealth at that exact moment = (2 * Debt) - Debt - Taxes
        margin_call_trigger_price = debt / (0.5 * total_shares)
        tax_at_margin_call = calculate_taxes(total_shares, margin_call_trigger_price, fmv_on_exercise, d)['tax_liability']
        net_wealth_at_margin_call = (2 * debt) - debt - tax_at_margin_call

        data.append({
            "Date": current_date,
            "Base Net Wealth": get_true_net_wealth(price_base),
            "Bull Net Wealth": get_true_net_wealth(price_bull),
            "Bear Net Wealth": get_true_net_wealth(price_bear),
            "Margin Call Threshold": max(0, net_wealth_at_margin_call)
        })
        
    return pd.DataFrame(data).set_index("Date")

# ==========================================
# NODE 4: THE UNBIASED QUANTITATIVE AGENT
# ==========================================
def generate_ai_insights(user_query: str, emp_status: str, total_shares: int, debt: float, market_data: dict, strategy: dict, margin_call: float, tax_data: dict, is_default: bool = False):
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key: return "⚠️ Error: Gemini API Key not found."
    
    trading_constraint = "SUBJECT TO STRICT 20-DAY QUARTERLY TRADING WINDOWS." if emp_status == "Active Employee" else "NO TRADING WINDOW RESTRICTIONS (Ex-Employee)."
    
    system_instruction = """
    You are an elite, UNBIASED quantitative wealth advisor. 
    YOUR DIRECTIVE: You MUST evaluate if holding the debt is mathematically superior to clearing it. 
    Compare the Nuvama interest rate (Max 9.25% p.a.) against the expected Analyst Target Yields and the STCG vs LTCG tax delta. 
    
    CRITICAL MARGIN CALL KNOWLEDGE: 
    1. The margin call triggers when Total Debt exceeds 50% of the Pledged Portfolio Value.
    2. If triggered, the user has a STRICT 7-DAY CURE PERIOD to regularize the account. 
    3. Do NOT suggest immediate panic selling on a margin call. Advise utilizing the 7-day window to wait for price recovery or injecting external short-term cash to drop the LTV back under 50%.
    
    Be extremely concise. Use short bullet points. Provide pure, actionable financial strategy.
    """
    
    state_context = f"""
    --- FINANCIAL STATE & PREDICTIONS ---
    Status: {emp_status} | Rules: {trading_constraint}
    Total Shares: {total_shares:,} | Current Loan Debt: ₹{debt:,.2f}
    Current Price: ₹{market_data['current_price']:,.2f} | 20-Day SMA: ₹{market_data['sma_20']:,.2f}
    Analyst 1Yr Targets -> Bull: ₹{market_data['bull_target']:,.2f} | Base: ₹{market_data['base_target']:,.2f} | Bear: ₹{market_data['bear_target']:,.2f}
    Current Tax Trigger: {tax_data['tax_type']}
    50% LTV Margin Call Price: ₹{margin_call:,.2f}
    """
    
    if is_default:
        prompt = f"{system_instruction}\n{state_context}\nTake a step back. Provide an unbiased baseline strategy. State clearly whether the optimal quantitative play is to HOLD or SELL, and factor in the 7-day 50% LTV margin call reality into your risk assessment."
    else:
        prompt = f"{system_instruction}\n{state_context}\nThe user asks: '{user_query}'. Refine their strategy."
    
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

# --- SIDEBAR ---
st.sidebar.header("⚙️ Portfolio Configuration")
ticker = st.sidebar.text_input("Live Ticker Symbol", value="MEESHO.NS")
emp_status = st.sidebar.radio("Employment Status", ["Active Employee", "Ex-Employee"])

st.sidebar.divider()
st.sidebar.subheader("🧮 Loan & Equity Calculator")

vested_options = st.sidebar.number_input("Total Vested Options (ESOPs)", min_value=0, value=0, step=100)
total_shares = st.sidebar.number_input("Total Resulting Shares", min_value=0, value=0, step=100)
fmv_on_exercise = st.sidebar.number_input("FMV on Exercise Date (₹)", min_value=0.0, value=130.0, step=1.0)

if 'principal_loan' not in st.session_state: st.session_state.principal_loan = 0.0
if 'calc_breakdown' not in st.session_state: st.session_state.calc_breakdown = ""

if st.sidebar.button("Calculate Exact Loan Needed"):
    if vested_options > 0 and total_shares > 0 and fmv_on_exercise > 0:
        exercise_payable = vested_options * 1
        perq_value = (total_shares * fmv_on_exercise) - exercise_payable
        tax_rate = 0.30 if emp_status == "Active Employee" else 0.4274
        perq_tax = perq_value * tax_rate
        
        st.session_state.principal_loan = round(exercise_payable + perq_tax, 2)
        st.session_state.calc_breakdown = f"**Calculation Breakdown:**\n- Exercise Price: ₹{exercise_payable:,.2f}\n- Perquisite Value: ₹{perq_value:,.2f}\n- Perquisite Tax ({tax_rate*100}%): ₹{perq_tax:,.2f}\n- **Total Loan: ₹{st.session_state.principal_loan:,.2f}**"
    else:
        st.sidebar.warning("Enter Options, Shares, and FMV first.")

if st.session_state.calc_breakdown: st.sidebar.info(st.session_state.calc_breakdown)

principal_loan = st.sidebar.number_input("Total Loan Principal (₹)", min_value=0.0, value=float(st.session_state.principal_loan), step=10000.0)

loan_sanction_date = st.sidebar.date_input("Loan Sanction Date", datetime.date.today() - datetime.timedelta(days=1))
days_held = max(1, (datetime.date.today() - loan_sanction_date).days)
st.sidebar.caption(f"Days Loan Held: **{days_held} days**")
sim_days = st.sidebar.slider("Simulate Future Date (Days Held)", min_value=1, max_value=400, value=days_held)

st.sidebar.divider()
st.sidebar.subheader("💸 Cash Injections (Multiple)")
if 'prepayments_df' not in st.session_state: st.session_state.prepayments_df = pd.DataFrame({"Day": [15], "Amount (₹)": [0]})
edited_prepayments = st.sidebar.data_editor(st.session_state.prepayments_df, num_rows="dynamic", hide_index=True)
prepayments_list = list(zip(edited_prepayments["Day"], edited_prepayments["Amount (₹)"]))

# --- MAIN DASHBOARD ---
if total_shares == 0 or principal_loan == 0:
    st.info("👋 Welcome. Please enter your Equity details and calculate your Loan Principal in the sidebar to begin.")
else:
    todays_debt = calculate_nuvama_debt(principal_loan, sim_days, prepayments_list)
    market_data = get_market_data(ticker)
    live_price = market_data['current_price']

    if live_price > 0:
        strategy = calculate_liquidation_strategy(todays_debt, live_price, total_shares)
        
        # 50% LTV Margin Call Logic Update
        danger_price = todays_debt / (total_shares * 0.5)
        
        st.header(f"Simulated Execution Status (Day {sim_days})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Live Share Price", f"₹{live_price:,.2f}")
        col2.metric("Simulated Debt", f"₹{todays_debt:,.2f}")
        col3.metric("Net Pocketed / Share", f"₹{strategy['net_pocketed']:,.2f}")
        
        st.divider()
        col4, col5 = st.columns(2)
        col4.metric("Shares to Sell to Clear Debt", f"{strategy['shares_to_sell']:,}", delta_color="inverse")
        col5.metric("Remaining Shares (Free & Clear)", f"{strategy['remaining_shares']:,}", f"Gross Value: ₹{strategy['unlocked_wealth']:,.2f}", delta_color="normal")
        
        st.warning(f"🚨 **50% LTV Margin Call Threshold:** ₹{danger_price:,.2f} per share. (If triggered, you have 7 days to cure the margin before forced liquidation).")

        st.divider()
        st.subheader("📊 True Net Wealth Projection (Post-Tax & Debt)")
        st.caption(f"This model maps the **50% LTV Margin Call Threshold**. If any portfolio line touches the red Margin Call line, your 7-day cure window begins.")
        
        proj_df = generate_projection_data(principal_loan, total_shares, fmv_on_exercise, market_data, prepayments_list, loan_sanction_date)
        st.line_chart(proj_df, color=["#0068C9", "#29B09D", "#FF8700", "#FF0000"]) 
        
        st.divider()
        st.subheader("🧠 Unbiased Quantitative Execution Plan")
        tax_data = calculate_taxes(strategy['shares_to_sell'], live_price, fmv_on_exercise, sim_days)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            with st.spinner("Computing unbiased baseline strategy..."):
                baseline_plan = generate_ai_insights("", emp_status, total_shares, todays_debt, market_data, strategy, danger_price, tax_data, is_default=True)
                st.session_state.messages.append({"role": "assistant", "content": f"**Unbiased Baseline Strategy Generated:**\n\n{baseline_plan}"})
                
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Refine this strategy (e.g., 'If a margin call triggers tomorrow, what is the optimal 7-day cure plan?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Recalculating strategy..."):
                    response = generate_ai_insights(prompt, emp_status, total_shares, todays_debt, market_data, strategy, danger_price, tax_data, is_default=False)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Could not fetch live market data. Please check the ticker symbol.")
