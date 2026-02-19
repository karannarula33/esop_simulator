import math
import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime
import plotly.express as px
from google import genai
from google.genai import types

# ==========================================
# NODE 0: FUNDING PARTNER TERMS DICTIONARY
# ==========================================
FUNDING_PARTNERS = {
    "Nuvama Wealth": {
        "doc_fee": 500,
        "processing_fee_pct": 0.0025,
        "margin_ltv": 0.50, 
        "cure_period_days": 7,
        "interest_tiers": [(30, 0.075), (60, 0.085), (400, 0.0925)] 
    },
    "Bajaj Financial Securities": {
        "doc_fee": 0,
        "processing_fee_pct": 0.0020,
        "margin_ltv": 0.50, # Standard fallback LTV
        "cure_period_days": 7, # Standard fallback cure period
        "interest_tiers": [(30, 0.0725), (90, 0.085), (400, 0.0925)]
    },
    "Infina Finance": {
        "doc_fee": 999,
        "processing_fee_pct": 0.0025,
        "margin_ltv": 0.50, 
        "cure_period_days": 5,
        "interest_tiers": [(30, 0.0825), (90, 0.0875), (180, 0.0950), (400, 0.10)] 
    },
    "360 ONE Prime": {
        "doc_fee": 0,
        "processing_fee_pct": 0.0015, 
        "margin_ltv": 0.40,
        "cure_period_days": 1,
        "interest_tiers": [(90, 0.095), (180, 0.10), (400, 0.105)]
    }
}

# ==========================================
# NODE 1: THE DYNAMIC DEBT ENGINE
# ==========================================
def calculate_loan_debt(principal: float, days_elapsed: int, prepayments: list, terms: dict) -> float:
    if principal <= 0: return 0.0
    
    processing_fee_base = principal * terms["processing_fee_pct"]
    processing_fee_gst = processing_fee_base * 0.18 
    total_fees_at_closure = terms["doc_fee"] + processing_fee_base + processing_fee_gst
    
    total_interest = 0.0
    current_principal = principal
    
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
# NODE 1.5: ADVANCED MARKET & DATA PROVENANCE
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
            volatility = min(daily_returns.std() * math.sqrt(252), 0.30)
        else:
            sma_20 = current_price
            volatility = 0.25 
            
        has_analyst_data = 'targetMeanPrice' in info and info['targetMeanPrice'] is not None
        
        bull_target = info.get('targetHighPrice') if has_analyst_data else current_price * 1.25
        base_target = info.get('targetMeanPrice') if has_analyst_data else current_price * 1.10
        bear_target = info.get('targetLowPrice') if has_analyst_data else current_price * 0.85
        
        return {
            "current_price": round(current_price, 2),
            "bull_target": round(bull_target, 2),
            "base_target": round(base_target, 2),
            "bear_target": round(bear_target, 2),
            "sma_20": round(sma_20, 2),
            "volatility": float(volatility),
            "has_analyst_data": has_analyst_data
        }
    except:
        return {"current_price": 0.0, "bull_target": 0.0, "base_target": 0.0, "bear_target": 0.0, "sma_20": 0.0, "volatility": 0.25, "has_analyst_data": False}

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
# NODE 2.5: MEAN-REVERTING STOCHASTIC PROJECTION 
# ==========================================
def generate_projection_data(principal: float, total_shares: int, fmv_on_exercise: float, market_data: dict, prepayments: list, sanction_date: datetime.date, terms: dict):
    np.random.seed(42) 
    days_to_sim = list(range(1, 400, 3)) 
    
    wealth_data = []
    price_data = []
    
    p0 = market_data['current_price']
    target = market_data['base_target']
    vol = market_data['volatility']
    
    current_sim_price = p0
    daily_drift = math.log(target / p0) / 365
    
    for d in days_to_sim:
        current_date = sanction_date + datetime.timedelta(days=d)
        debt = calculate_loan_debt(principal, d, prepayments, terms)
        
        if d > 1:
            step_vol = vol * math.sqrt(3/252)
            z = np.random.normal(0, 1)
            linear_expectation = p0 + ((target - p0) / 365) * d
            reversion = (linear_expectation - current_sim_price) * 0.08
            current_sim_price = current_sim_price * math.exp(daily_drift * 3 - 0.5 * step_vol**2 + step_vol * z) + reversion
            
        gross_value = current_sim_price * total_shares
        
        # Margin call level is dynamically driven by the specific partner's LTV rules
        margin_call_level = debt / terms["margin_ltv"]
        
        tax_hit = calculate_taxes(total_shares, current_sim_price, fmv_on_exercise, d)['tax_liability']
        net_wealth = max(0, gross_value - debt - tax_hit)

        formatted_price = f"₹{current_sim_price:,.2f}"

        wealth_data.append({
            "Date": current_date,
            "Gross Portfolio Value (₹)": gross_value,
            "Net Wealth (₹)": net_wealth,
            "Margin Call Threshold (₹)": margin_call_level,
            "Total Debt (₹)": debt,
            "Underlying Share Price": formatted_price
        })
        
        price_data.append({
            "Date": current_date,
            "Simulated Price (₹)": current_sim_price,
            "Bull Target (₹)": market_data['bull_target'],
            "Base Target (₹)": market_data['base_target'],
            "Bear Target (₹)": market_data['bear_target']
        })
        
    return pd.DataFrame(wealth_data).set_index("Date"), pd.DataFrame(price_data).set_index("Date")

# ==========================================
# NODE 4: THE INTERACTIVE QUANTITATIVE AGENT
# ==========================================
def generate_ai_insights(user_query: str, emp_status: str, total_shares: int, debt: float, market_data: dict, strategy: dict, margin_call: float, tax_data: dict, days_held: int, partner_name: str, terms: dict, is_default: bool = False):
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key: return "⚠️ Error: Gemini API Key not found."
    
    trading_constraint = "CRITICAL: The user is an Active Employee. ALL stock sales MUST be explicitly scheduled during the '20-day quarterly trading windows' that open strictly 48 hours after financial results are published. Do not suggest selling outside these windows." if emp_status == "Active Employee" else "The user is an Ex-Employee. There are no trading window blackout restrictions. They can execute limit orders on any trading day."
    target_confidence = "VERIFIED INSTITUTIONAL DATA" if market_data['has_analyst_data'] else "ALGORITHMIC FALLBACK (LOW CONFIDENCE)"
    
    state_context = f"""
    --- EXACT FINANCIAL STATE ---
    Funding Partner: {partner_name}
    Partner Margin Call LTV: {terms['margin_ltv']*100}%
    Partner Cure Period: {terms['cure_period_days']} days
    Total Shares: {total_shares:,} | Current Loan Debt: ₹{debt:,.2f} | Days Loan Held: {days_held}
    Current Price: ₹{market_data['current_price']:,.2f} | 20-Day SMA: ₹{market_data['sma_20']:,.2f}
    Target Confidence: {target_confidence}
    1Yr Targets -> Bull: ₹{market_data['bull_target']:,.2f} | Base: ₹{market_data['base_target']:,.2f} | Bear: ₹{market_data['bear_target']:,.2f}
    Current Tax Trigger: {tax_data['tax_type']} (Liability: ₹{tax_data['tax_liability']:,.2f})
    Margin Call Price: ₹{margin_call:,.2f}
    Minimum Shares to clear debt today: {strategy.get('shares_to_sell', 0):,}
    """
    
    if is_default:
        prompt = f"""
        You are an elite quantitative wealth advisor. 
        Evaluate the following state:
        {state_context}
        Rules: {trading_constraint}
        
        DO NOT GENERATE A PHASED SCHEDULE YET.
        
        Instead, output a highly concise "Portfolio Health Check" containing:
        1. A 1-sentence assessment of their Margin Call Risk (Factoring in the exact partner LTV and cure period).
        2. A 1-sentence assessment of their Tax State (STCG vs LTCG).
        
        Then, end your response by asking the user a multiple-choice question to determine their primary strategic intent so you can build the right schedule. Provide these 3 options:
        [A] Aggressive Debt Elimination (I want to be debt-free ASAP)
        [B] Maximize Long-Term Wealth (I am willing to hold the loan to get LTCG tax benefits and stock upside)
        [C] External Capital Extraction (I need to liquidate a specific amount for a startup, real estate, etc.)
        
        Keep this incredibly brief.
        """
    else:
        prompt = f"""
        You are an elite, algorithmic execution engine.
        {state_context}
        Rules: {trading_constraint}
        
        The user replied with: "{user_query}"
        
        Based on their chosen intent, generate a mathematically precise, multi-tranche execution schedule. 
        Do not provide vague advice. Calculate exact share quantities and explicit target prices based on the specific partner's interest rates and cure terms.
        
        Format your response as:
        **Phase 1: [Name tailored to their goal]**
        * Action: [Sell X / Hold]
        * Timing: [Specific Timing factoring in Trading Windows and the 365-day LTCG cliff]
        * Target Price: [Rupee Value]
        
        **Phase 2: [Name tailored to their goal]**
        ... etc.
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

# --- SIDEBAR ---
st.sidebar.header("⚙️ Portfolio Configuration")
ticker = st.sidebar.text_input("Live Ticker Symbol", value="MEESHO.NS")
emp_status = st.sidebar.radio("Employment Status", ["Active Employee", "Ex-Employee"])

st.sidebar.divider()
st.sidebar.subheader("🏦 Funding Partner Selection")
selected_partner = st.sidebar.selectbox("Select your financer:", list(FUNDING_PARTNERS.keys()))
active_terms = FUNDING_PARTNERS[selected_partner]

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
    todays_debt = calculate_loan_debt(principal_loan, sim_days, prepayments_list, active_terms)
    market_data = get_market_data(ticker)
    live_price = market_data['current_price']

    if live_price > 0:
        strategy = calculate_liquidation_strategy(todays_debt, live_price, total_shares)
        
        danger_price = todays_debt / (total_shares * active_terms['margin_ltv'])
        
        st.header(f"Simulated Execution Status (Day {sim_days})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Live Share Price", f"₹{live_price:,.2f}")
        col2.metric("Simulated Debt", f"₹{todays_debt:,.2f}")
        col3.metric("Net Pocketed / Share", f"₹{strategy['net_pocketed']:,.2f}")
        
        st.divider()
        col4, col5 = st.columns(2)
        col4.metric("Shares to Sell to Clear Debt", f"{strategy['shares_to_sell']:,}", delta_color="inverse")
        col5.metric("Remaining Shares (Free & Clear)", f"{strategy['remaining_shares']:,}", f"Gross Value: ₹{strategy['unlocked_wealth']:,.2f}", delta_color="normal")
        
        st.warning(f"🚨 **{int(active_terms['margin_ltv']*100)}% LTV Margin Call Threshold ({selected_partner}):** ₹{danger_price:,.2f} per share. (If your stock drops to this price, your loan LTV hits {int(active_terms['margin_ltv']*100)}%. You then have a strict {active_terms['cure_period_days']}-day cure window to inject cash or sell before forced liquidation).")

        wealth_df, price_df = generate_projection_data(principal_loan, total_shares, fmv_on_exercise, market_data, prepayments_list, loan_sanction_date, active_terms)
        
        st.divider()
        st.subheader("📈 Projected Share Price vs. Analyst Benchmarks")
        
        if market_data['has_analyst_data']:
            st.success("✅ **Data Provenance:** Target benchmarks are pulled from live Wall Street Analyst Consensus.")
        else:
            st.warning("⚠️ **Data Provenance:** Algorithmic Fallback. Live analyst consensus targets are unavailable for this ticker. Targets shown are simulated (+25% / +10% / -15%).")
            
        fig_price = px.line(
            price_df.reset_index(), 
            x="Date", 
            y=["Bull Target (₹)", "Base Target (₹)", "Bear Target (₹)", "Simulated Price (₹)"],
            color_discrete_sequence=["#29B09D", "#7C3AED", "#FF4B4B", "#0068C9"]
        )
        fig_price.update_traces(hovertemplate="₹%{y:,.2f}")
        fig_price.update_layout(hovermode="x unified", xaxis_title="", yaxis_title="Share Price (₹)", legend_title="")
        st.plotly_chart(fig_price, use_container_width=True)
        
        st.subheader("📊 True Net Wealth Trendline (Post-Tax & Debt)")
        st.caption("Notice the vertical 'step up' at month 12 when your tax burden drops to the 12.5% LTCG rate.")
        
        fig_wealth = px.line(
            wealth_df.reset_index(), 
            x="Date", 
            y=["Gross Portfolio Value (₹)", "Net Wealth (₹)", "Margin Call Threshold (₹)", "Total Debt (₹)"],
            color_discrete_sequence=["#0068C9", "#29B09D", "#FF8700", "#FF4B4B"],
            custom_data=["Underlying Share Price"]
        )
        fig_wealth.update_traces(hovertemplate="₹%{y:,.2f}  (Share Price: %{customdata[0]})")
        fig_wealth.update_layout(hovermode="x unified", xaxis_title="", yaxis_title="Total Value (₹)", legend_title="")
        st.plotly_chart(fig_wealth, use_container_width=True)
        
        # --- BOTTOM SECTION: INLINE CHAT ---
        st.divider()
        st.subheader("🧠 Interactive Algorithmic Execution Agent")
        tax_data = calculate_taxes(strategy['shares_to_sell'], live_price, fmv_on_exercise, sim_days)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            with st.spinner("Analyzing portfolio health..."):
                baseline_plan = generate_ai_insights("", emp_status, total_shares, todays_debt, market_data, strategy, danger_price, tax_data, sim_days, selected_partner, active_terms, is_default=True)
                st.session_state.messages.append({"role": "assistant", "content": f"{baseline_plan}"})
                
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.form("strategy_chat_form", clear_on_submit=True):
            user_prompt = st.text_input("Reply with A, B, C, or type a custom goal:", placeholder="e.g., 'C - I need ₹20 Lakhs in exactly 4 months.'")
            submitted = st.form_submit_button("Generate Strategic Schedule")
            
        if submitted and user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.spinner("Calculating precision tranches based on your objective..."):
                response = generate_ai_insights(user_prompt, emp_status, total_shares, todays_debt, market_data, strategy, danger_price, tax_data, sim_days, selected_partner, active_terms, is_default=False)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    else:
        st.error("Could not fetch live market data. Please check the ticker symbol.")
