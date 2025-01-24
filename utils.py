import pandas as pd
import json


def update_df_style(df):
    df = df.copy()
    df.loc["Total"] = df.sum(numeric_only=True)
    # Remove NA as Empty String
    df = df.fillna("")
    # Add proper number formatting if columns are numeric
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].map("{:,.0f}".format)

    return df


# Calculate target amounts dynamically based on additional investment
def calculate_additional_investment(
    existing_balances, investable_amount, allocations, recompute_till_non_zero=False
):
    # Start with no additional investment
    additional_investment = 0

    while True:
        # Total available funds (existing + investable + additional)
        total_funds = (
            sum(existing_balances.values()) + investable_amount + additional_investment
        )

        # Calculate target amounts based on allocations
        target_amounts = {
            asset: round((percent / 100) * total_funds, 2)
            for asset, percent in allocations.items()
        }

        # Calculate rebalancing amounts
        rebalancing_amounts = {
            asset: round(target_amounts[asset] - balance, 2)
            for asset, balance in existing_balances.items()
        }

        if not recompute_till_non_zero:
            return additional_investment, target_amounts, rebalancing_amounts

        # Check if all rebalancing amounts are non-negative
        if all(amount >= 0 for amount in rebalancing_amounts.values()):
            return additional_investment, target_amounts, rebalancing_amounts

        # Increment additional investment to eliminate negative values
        additional_investment += 1000  # Increment by ₹1,000 (adjust as needed)


def get_default_allocation(risk_profile, only_equity):
    if only_equity:
        # Only Equity Allocation based on Risk Profile
        if risk_profile == "High Risk":
            return {
                "Large Cap": 40,
                "Mid Cap": 40,
                "Small Cap": 20,
                "Debt": 0,
                "Gold": 0,
            }
        elif risk_profile == "Moderate Risk":
            return {
                "Large Cap": 50,
                "Mid Cap": 30,
                "Small Cap": 20,
                "Debt": 0,
                "Gold": 0,
            }
        else:  # Low Risk
            return {
                "Large Cap": 70,
                "Mid Cap": 20,
                "Small Cap": 10,
                "Debt": 0,
                "Gold": 0,
            }
    else:

        if risk_profile == "High Risk":
            return {
                "Large Cap": 40,
                "Mid Cap": 30,
                "Small Cap": 20,
                "Debt": 5,
                "Gold": 5,
            }
        elif risk_profile == "Moderate Risk":
            return {
                "Large Cap": 50,
                "Mid Cap": 25,
                "Small Cap": 10,
                "Debt": 10,
                "Gold": 5,
            }
        elif risk_profile == "Only Equity":
            return {
                "Large Cap": 60,
                "Mid Cap": 25,
                "Small Cap": 15,
                "Debt": 0,
                "Gold": 0,
            }
        else:  # Low Risk
            return {
                "Large Cap": 30,
                "Mid Cap": 10,
                "Small Cap": 5,
                "Debt": 50,
                "Gold": 5,
            }


def load_user_data():
    try:
        with open("user_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_user_data(data):
    with open("user_data.json", "w") as f:
        json.dump(data, f, indent=4)


def load_user_profile(user_id):
    user_data = load_user_data()
    return user_data.get(user_id, {})


def save_user_profile(user_id, profile):
    if not user_id:
        return False
    user_data = load_user_data()
    user_data[user_id] = profile
    save_user_data(user_data)
    return True
