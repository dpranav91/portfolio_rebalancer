# app.py
import streamlit as st
import pandas as pd

# Application Title
st.title("Mutual Fund Portfolio Rebalancer with Existing Balances")

# Initialize session state for inputs
if "total_investable" not in st.session_state:
    st.session_state.total_investable = 100000.0
if "existing_balances" not in st.session_state:
    st.session_state.existing_balances = {
        "Large Cap": 0.0,
        "Mid Cap": 0.0,
        "Small Cap": 0.0,
        "Gold": 0.0,
        "Debt": 0.0,
    }
if "allocations" not in st.session_state:
    st.session_state.allocations = {
        "Large Cap": 30,
        "Mid Cap": 20,
        "Small Cap": 10,
        "Gold": 10,
        "Debt": 30,
    }

# Sidebar Inputs
st.sidebar.header("Portfolio Inputs")
st.session_state.total_investable = st.sidebar.number_input("Enter Total Investable Amount (₹):", min_value=0.0, step=1000.0, value=st.session_state.total_investable)

# Input existing balances for each asset class
st.sidebar.subheader("Enter Existing Balances (₹):")
st.session_state.existing_balances["Large Cap"] = st.sidebar.number_input("Large Cap Balance (₹):", min_value=0.0, step=100.0, value=st.session_state.existing_balances["Large Cap"])
st.session_state.existing_balances["Mid Cap"] = st.sidebar.number_input("Mid Cap Balance (₹):", min_value=0.0, step=100.0, value=st.session_state.existing_balances["Mid Cap"])
st.session_state.existing_balances["Small Cap"] = st.sidebar.number_input("Small Cap Balance (₹):", min_value=0.0, step=100.0, value=st.session_state.existing_balances["Small Cap"])
st.session_state.existing_balances["Gold"] = st.sidebar.number_input("Gold Balance (₹):", min_value=0.0, step=100.0, value=st.session_state.existing_balances["Gold"])
st.session_state.existing_balances["Debt"] = st.sidebar.number_input("Debt Balance (₹):", min_value=0.0, step=100.0, value=st.session_state.existing_balances["Debt"])

# Input target allocation percentages
st.sidebar.subheader("Set Target Allocation Percentages:")
st.session_state.allocations["Large Cap"] = st.sidebar.slider("Large Cap (%)", 0, 100, st.session_state.allocations["Large Cap"])
st.session_state.allocations["Mid Cap"] = st.sidebar.slider("Mid Cap (%)", 0, 100, st.session_state.allocations["Mid Cap"])
st.session_state.allocations["Small Cap"] = st.sidebar.slider("Small Cap (%)", 0, 100, st.session_state.allocations["Small Cap"])
st.session_state.allocations["Gold"] = st.sidebar.slider("Gold (%)", 0, 100, st.session_state.allocations["Gold"])
st.session_state.allocations["Debt"] = st.sidebar.slider("Debt (%)", 0, 100, st.session_state.allocations["Debt"])

# Ensure total allocation equals 100%
total_allocation = sum(st.session_state.allocations.values())
if total_allocation != 100:
    st.warning(f"Total allocation is {total_allocation}%. Adjusting sliders automatically to sum up to 100%.")
    st.session_state.allocations = {key: round(value / total_allocation * 100) for key, value in st.session_state.allocations.items()}

# Display current balances and target allocations
st.subheader("Current Balances and Target Allocations")
current_df = pd.DataFrame({
    "Asset Class": st.session_state.existing_balances.keys(),
    "Existing Balance (₹)": st.session_state.existing_balances.values(),
    "Target Allocation (%)": st.session_state.allocations.values(),
})
current_df.loc["Total"] = current_df.sum(numeric_only=True)
current_df = current_df.style.format("{:,.0f}", subset=["Existing Balance (₹)", "Target Allocation (%)"])

st.table(current_df)

# Calculate the total current balance and the total target amount
current_total_balance = sum(st.session_state.existing_balances.values())
total_target_amount = st.session_state.total_investable + current_total_balance

# Calculate the target amounts and rebalancing amounts
target_amounts = {asset: round((percent / 100) * total_target_amount, 2) for asset, percent in st.session_state.allocations.items()}
st.write("**Target Amounts (₹):**", target_amounts)
rebalancing_amounts = {asset: round(target_amounts[asset] - balance, 2) for asset, balance in st.session_state.existing_balances.items()}
st.write("**Rebalancing Amounts (₹):**", rebalancing_amounts)

# Recommend additional investment if there are negative rebalancing amounts
negative_rebalance = {asset: amount for asset, amount in rebalancing_amounts.items() if amount < 0}
if negative_rebalance:
    additional_investment_needed = abs(sum(negative_rebalance.values()))
    st.warning(f"Additional investment of ₹{additional_investment_needed} is recommended to eliminate negative values in the rebalanced portfolio.")

# Display rebalancing results
st.subheader("Rebalanced Portfolio")
rebalance_df = pd.DataFrame({
    "Asset Class": target_amounts.keys(),
    "Target Amount (₹)": target_amounts.values(),
    "Existing Balance (₹)": st.session_state.existing_balances.values(),
    "To Be Invested/Rebalanced (₹)": rebalancing_amounts.values(),
})
# Add Total row
rebalance_df.loc["Total"] = rebalance_df.sum(numeric_only=True)

rebalance_df = rebalance_df.style.format("{:,.0f}", subset=["Target Amount (₹)", "Existing Balance (₹)", "To Be Invested/Rebalanced (₹)"])
st.table(rebalance_df)

# Summary
total_invested = sum(target_amounts.values())
remaining_cash = round(st.session_state.total_investable - sum(rebalancing_amounts.values()), 2)
st.write(f"**Total Investable Amount:** ₹{st.session_state.total_investable}")
st.write(f"**Total Invested:** ₹{total_invested}")
st.write(f"**Remaining Cash:** ₹{remaining_cash}")

# Deployment Instructions
st.info("To deploy this app locally: Run `streamlit run app.py` in your terminal.")
