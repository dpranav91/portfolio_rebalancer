import pandas as pd

def update_df_style(df):
    df = df.copy()
    df.loc["Total"] = df.sum(numeric_only=True)
    # Add proper number formatting if columns are numeric
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].map("{:,.0f}".format)
    
    return df


# Calculate target amounts dynamically based on additional investment
def calculate_additional_investment(existing_balances, investable_amount, allocations, recompute_till_non_zero=False):
    # Start with no additional investment
    additional_investment = 0

    while True:
        # Total available funds (existing + investable + additional)
        total_funds = sum(existing_balances.values()) + investable_amount + additional_investment

        # Calculate target amounts based on allocations
        target_amounts = {asset: round((percent / 100) * total_funds, 2) for asset, percent in allocations.items()}

        # Calculate rebalancing amounts
        rebalancing_amounts = {asset: round(target_amounts[asset] - balance, 2) for asset, balance in existing_balances.items()}

        if not recompute_till_non_zero:
            return additional_investment, target_amounts, rebalancing_amounts
        
        # Check if all rebalancing amounts are non-negative
        if all(amount >= 0 for amount in rebalancing_amounts.values()):
            return additional_investment, target_amounts, rebalancing_amounts

        # Increment additional investment to eliminate negative values
        additional_investment += 1000  # Increment by ₹1,000 (adjust as needed)
