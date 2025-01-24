import streamlit as st
import pandas as pd

from utils import update_df_style, calculate_additional_investment



# -----------------
# Sidebar
# -----------------


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
st.session_state.total_investable = st.sidebar.number_input(
    "Enter Total Investable Amount (₹):", min_value=0.0, step=1000.0, value=st.session_state.total_investable
)

# Horizontal rule for better separation
st.sidebar.markdown("---")


# Input existing balances for each asset class
st.sidebar.subheader("Enter Existing Balances (₹):")
for asset in st.session_state.existing_balances.keys():
    st.session_state.existing_balances[asset] = st.sidebar.number_input(
        f"{asset} Balance (₹):", min_value=0.0, step=100.0, value=st.session_state.existing_balances[asset]
    )

# Horizontal rule for better separation
st.sidebar.markdown("---")

# Input target allocation percentages
st.sidebar.subheader("Set Target Allocation Percentages:")

for asset in st.session_state.allocations.keys():
    # Input number box for setting the allocation percentage
    st.session_state.allocations[asset] = st.sidebar.number_input(
        f"{asset} Allocation (%)", min_value=0, max_value=100, value=st.session_state.allocations[asset], step=1
    )

# Ensure total allocation equals 100%
total_allocation = sum(st.session_state.allocations.values())
if total_allocation != 100:
    st.sidebar.warning(
        f"Total allocation is {total_allocation}%."
    )
    st.sidebar.warning(f"Adjust remaining {100 - total_allocation}% to equal 100%.")
    st.sidebar.warning("Recommended allocations are displayed below.")

# Horizontal rule for better separation
st.sidebar.markdown("---")

# Recommended Allocations
st.sidebar.header("Allocations For Reference")
recommended_allocations = {key: round(value / total_allocation * 100) for key, value in st.session_state.allocations.items()}
for asset in recommended_allocations.keys():
    # Slider for reference (linked to the same value)
    st.sidebar.slider(
        f"{asset} %", 
        min_value=0, 
        max_value=100, 
        value=recommended_allocations[asset], 
        disabled=True  # Make the slider read-only
    )



# -----------------
# Main Panel
# -----------------


# Application Title
st.title("Portfolio Rebalancer")


# =====================
# Current Balances
# =====================

# Display current balances and target allocations
st.subheader("Current Balances and Target Allocations")
current_df = pd.DataFrame({
    "Asset Class": st.session_state.existing_balances.keys(),
    "Existing Balance (₹)": st.session_state.existing_balances.values(),
    "Target Allocation (%)": st.session_state.allocations.values(),
})
st.table(update_df_style(current_df))

# Perform calculation to find exact additional investment needed
additional_investment_needed, target_amounts, rebalancing_amounts = calculate_additional_investment(
    st.session_state.existing_balances,
    st.session_state.total_investable,
    st.session_state.allocations,
    recompute_till_non_zero=False
)


# =====================
# Rebalancing
# =====================

# Display rebalancing results
st.subheader("Rebalanced Portfolio")
rebalance_df = pd.DataFrame({
    "Asset Class": target_amounts.keys(),
    "Target Amount (₹)": target_amounts.values(),
    "Existing Balance (₹)": st.session_state.existing_balances.values(),
    "To Be Invested/Rebalanced (₹)": rebalancing_amounts.values(),
})
st.table(update_df_style(rebalance_df))

amount_to_sell = sum(amount for amount in rebalancing_amounts.values() if amount < 0)
if amount_to_sell < 0:
    st.warning(f"You need to sell ₹{abs(amount_to_sell)} assets to rebalance.") 
else:
    st.success(f"Portfolio is balanced as per target allocations!")


# =====================
# Additional Investment
# =====================

# Check if additional investment is needed
has_negative_values = any(amt < 0 for amt in rebalancing_amounts.values())

if has_negative_values:
    st.subheader("Recommendation for Additional Investment")

    additional_investment_needed, target_amounts, rebalancing_amounts = calculate_additional_investment(
        st.session_state.existing_balances,
        st.session_state.total_investable,
        st.session_state.allocations,
        recompute_till_non_zero=True
    )   
    st.warning(f"Additional investment of ₹{additional_investment_needed:,.2f} is required to avoid selling any existing assets.")

    # Display rebalancing results
    st.subheader("Rebalanced Portfolio With New Additional Investment")
    rebalance_df = pd.DataFrame({
        "Asset Class": target_amounts.keys(),
        "Target Amount (₹)": target_amounts.values(),
        "Existing Balance (₹)": st.session_state.existing_balances.values(),
        "To Be Invested/Rebalanced (₹)": rebalancing_amounts.values(),
    })
    st.table(update_df_style(rebalance_df))

