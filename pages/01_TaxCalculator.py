import streamlit as st
import pandas as pd

# Wide layout
st.set_page_config(layout="wide", page_title="Income Tax Calculator", page_icon="💰")

LAKH = 100000
CRORE = 100 * LAKH

# Define tax slabs in Descending order
TAX_DATA = {
    "2025": {
        "slabs": [
            {"range": (0, 400000), "rate": 0},
            {"range": (400000, 800000), "rate": 5},
            {"range": (800000, 1200000), "rate": 10},
            {"range": (1200000, 1600000), "rate": 15},
            {"range": (1600000, 2000000), "rate": 20},
            {"range": (2000000, 2400000), "rate": 25},
            {"range": (2400000, 5000000), "rate": 30},
            {"range": (5000000, 1 * CRORE), "rate": 30, "surcharge": 10},
            {"range": (1 * CRORE, 2 * CRORE), "rate": 30, "surcharge": 15},
            {"range": (2 * CRORE, float("inf")), "rate": 30, "surcharge": 25},
        ],
        "standard_deduction": 75000,
        "tax_rebate_limit": 1200000,
    },
    "2024": {
        "slabs": [
            {"range": (0, 300000), "rate": 0},
            {"range": (300000, 700000), "rate": 5},
            {"range": (700000, 1000000), "rate": 10},
            {"range": (1000000, 1200000), "rate": 15},
            {"range": (1200000, 1500000), "rate": 20},
            {"range": (1500000, 5000000), "rate": 30},
            {"range": (5000000, 1 * CRORE), "rate": 30, "surcharge": 10},
            {"range": (1 * CRORE, 2 * CRORE), "rate": 30, "surcharge": 15},
            {"range": (2 * CRORE, float("inf")), "rate": 30, "surcharge": 25},
        ],
        "standard_deduction": 75000,
        "tax_rebate_limit": 700000,
    },
}

STANDARD_DEDUCTION = 75000


def number_to_words(num):
    """
    Converts a number into a human-readable format with suffixes like K, L, and CR.

    Args:
        num (int or float): The number to convert.

    Returns:
        str: The number converted into words with appropriate suffix.
    """
    if num < 1000:
        return str(num)
    elif num < 100000:  # Less than 1 lakh
        value = num / 1000
        suffix = "K"
    elif num < 10000000:  # Less than 1 crore
        value = num / 100000
        suffix = "L"
    else:  # 1 crore or more
        value = num / 10000000
        suffix = "CR"

    # Format the value to a maximum of 3 decimal places, removing unnecessary trailing zeros
    formatted_value = f"{value:.2f}".rstrip("0").rstrip(".")

    return f"{formatted_value} {suffix}"


def _append_suffix(value):
    if value < LAKH:
        return ""
    return f" ({number_to_words(value)})"


def calculate_tax_table(taxable_income, slabs, tax_rebate_limit=0):
    rows = []
    total_tax = surcharge = cess = cumulative_tax = final_tax = 0

    for slab in slabs:
        lower_limit, upper_limit = slab["range"]
        rate = slab["rate"]

        if taxable_income > lower_limit:
            taxable_amount = min(
                taxable_income - lower_limit, upper_limit - lower_limit
            )
            slab_tax = taxable_amount * rate / 100
            cumulative_tax += slab_tax

            # Apply surcharge if applicable
            if slabs[-1].get("surcharge") and taxable_income > slabs[-2]["range"][1]:
                surcharge = cumulative_tax * slabs[-1]["surcharge"] / 100

            # Apply health and education cess
            cess = (cumulative_tax + surcharge) * 4 / 100

            final_tax = cumulative_tax + surcharge + cess

            # Apply tax rebate if applicable
            tax_rebate = 0
            if tax_rebate_limit and taxable_income <= tax_rebate_limit:
                tax_rebate = -min(final_tax, tax_rebate_limit)
                final_tax = 0
                surcharge = cess = 0

            rows.append(
                {
                    "Income Level": f"₹{lower_limit:,} - ₹{upper_limit if upper_limit != float('inf') else taxable_income:,}",
                    "Slab Rate (%)": f"{rate}%",
                    "Total Tax for Slab (₹)": f"₹{slab_tax:,.0f}",
                    "Overall Tax Cumulative (₹)": f"₹{cumulative_tax:,.0f}",
                    "Rebate (₹)": tax_rebate,
                    "Surcharge (₹)": f"₹{surcharge:,.0f}",
                    "H/E Cess (₹)": f"₹{cess:,.0f}",
                    "Final Tax After Surcharge and Cess (₹)": f"₹{final_tax:,.0f}",
                }
            )

            if taxable_income <= upper_limit:
                break

    return rows


def tax_calculator_page():
    # Streamlit app
    st.title("Income Tax Calculator")

    # Input field for annual income
    income = st.number_input(
        "Enter your annual income",
        min_value=0.0,
        format="%.0f",
        value=1200000.0,
        step=50000.0,
    )

    # Deductions
    employer_nps_contribution = st.number_input(
        "Enter Annual Employer NPS contribution 80CCD(2) (₹)",
        min_value=0.0,
        format="%.0f",
        value=0.0,
        step=5000.0,
    )

    standard_deduction = st.number_input(
        "Standard Deduction (₹)", value=STANDARD_DEDUCTION
    )

    # Calculate Taxable Income
    taxable_income = max(0, income - employer_nps_contribution - standard_deduction)

    st.markdown(
        f"##### Taxable Income: ₹{taxable_income:,.0f}" + _append_suffix(taxable_income)
    )

    # Flag to check where to compare the tax benefit
    should_compare = st.checkbox("Compare Tax Benefit", value=False)

    # Store the tax calculation for each year
    result = {}  # {year: tax_liability}

    for year in TAX_DATA.keys():
        # Add a separator
        st.markdown("---")

        st.markdown(f"## Income Tax Calculation for {year} Tax Slabs")
        tax_data = TAX_DATA[year]
        slabs = tax_data["slabs"]
        tax_rebate_limit = tax_data.get("tax_rebate_limit", 0)
        standard_deduction = tax_data.get("standard_deduction", 0)

        # Calculate tax when button is clicked
        tax_table_data = calculate_tax_table(taxable_income, slabs, tax_rebate_limit)

        # Convert tax table data to DataFrame for display
        df = pd.DataFrame(tax_table_data)

        # Display the tax table
        st.markdown(f"#### Tax Calculation Table")
        st.dataframe(df)

        # Show final tax liability
        if tax_table_data:
            final_row = tax_table_data[-1]
            final_tax = final_row["Final Tax After Surcharge and Cess (₹)"]
            final_tax_float = float(final_tax.replace("₹", "").replace(",", ""))
            # Subheading for Tax Liability
            st.markdown("#### Tax Liability Under the New Tax Regime")

            # Calculate and display yearly tax liability
            result[year] = final_tax_float
            st.success(
                f"Yearly Tax Liability: ₹{final_tax_float:,.0f}"
                + _append_suffix(final_tax_float)
            )

            # Calculate and display monthly tax liability
            monthly_tax = final_tax_float / 12
            st.info(
                f"Monthly Tax Liability: ₹{monthly_tax:,.0f}"
                + _append_suffix(monthly_tax)
            )

        # Do not proceed if the flag to compare is not set
        if not should_compare:
            return

    # Add a separator
    st.markdown("---")

    # Show the Benefit
    st.write("## Tax Benefit Comparison")
    tax_years = list(TAX_DATA.keys())
    for index, year in enumerate(tax_years):
        if index == 0:
            continue
        next_year = tax_years[index - 1]
        benefit = result[year] - result[next_year]
        st.success(
            f"Your total tax benefit from {year} to {next_year} tax slab change is: ₹{benefit:,.0f}"
            + _append_suffix(benefit)
        )


tax_calculator_page()
