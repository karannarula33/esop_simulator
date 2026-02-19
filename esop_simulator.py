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
st.set_page_config(page_title="Smart ESOP Advisory",
