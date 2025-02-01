import streamlit as st
import pandas as pd

# Define tax slabs for 2024 and 2025
tax_slabs = {
    "2024": [
        {"range": (0, 400000), "rate": 0},
        {"range": (400001, 800000), "rate": 5},
        {"range": (800001, 1200000), "rate": 10},
        {"range": (1200001, 1600000), "rate": 15},
        {"range": (1600001, 2000000), "rate": 20},
        {"range": (2000001, 2400000), "rate": 25},
        {"range": (2400001, 5000000), "rate": 30},
        {"range": (5000001, float('inf')), "rate": 30, "surcharge": 30},
    ],
    "2025": [
        {"range": (0, 400000), "rate": 0},
        {"range": (400000, 800000), "rate": 5},
        {"range": (800000, 1200000), "rate": 10},
        {"range": (1200000, 1600000), "rate": 15},
        {"range": (1600000, 2000000), "rate": 20},
        {"range": (2000000, 2400000), "rate": 25},
        {"range": (2400000, 5000000), "rate": 30},
        {"range": (5000000, float('inf')), "rate": 30, "surcharge": 30},
    ],
}

STANDARD_DEDUCTION = 75000


def calculate_tax_table(taxable_income, slabs):
    rows = []
    total_tax = surcharge = cess = cumulative_tax = final_tax = 0

    for slab in slabs:
        lower_limit, upper_limit = slab["range"]
        rate = slab["rate"]

        if taxable_income > lower_limit:
            taxable_amount = min(taxable_income - lower_limit, upper_limit - lower_limit)
            slab_tax = taxable_amount * rate / 100
            cumulative_tax += slab_tax

            # Apply surcharge if applicable
            if slabs[-1].get("surcharge") and taxable_income > slabs[-2]["range"][1]:
                surcharge = cumulative_tax * slabs[-1]["surcharge"] / 100

            # Apply health and education cess
            cess = (cumulative_tax + surcharge) * 4 / 100
            final_tax = cumulative_tax + surcharge + cess

            rows.append({
                "Income Level": f"₹{lower_limit:,} - ₹{upper_limit if upper_limit != float('inf') else taxable_income:,}",
                "Slab Rate (%)": f"{rate}%",
                "Total Tax for Slab (₹)": f"₹{slab_tax:,.0f}",
                "Overall Tax Cumulative (₹)": f"₹{cumulative_tax:,.0f}",
                "Surcharge (₹)": f"₹{surcharge:,.0f}",
                "H/E Cess (₹)": f"₹{cess:,.0f}",
                "Final Tax After Surcharge and Cess (₹)": f"₹{final_tax:,.0f}"
            })

            if taxable_income <= upper_limit:
                break

    return rows


# Streamlit app
st.title("Income Tax Calculator with Table View")

# Dropdown to select the year
year = st.selectbox("Select Tax Slab Year", ["2025", "2024"], index=0)

# Input field for annual income
income = st.number_input("Enter your annual income", min_value=0.0, format="%.0f", value=1200000.0, step=50000.0)

# Deductions
employer_nps_contribution = st.number_input("Enter Annual Employer NPS contribution 80CCD(2) (₹)", min_value=0.0, format="%.0f", value=0.0, step=5000.0)

st.write(f"Standard Deduction: ₹{employer_nps_contribution:,.0f}")

# Apply standard deduction
taxable_income = max(0, income - employer_nps_contribution - employer_nps_contribution)

st.markdown(f"##### Taxable Income: ₹{taxable_income:,.0f}")

# Calculate tax when button is clicked
if 1 or st.button("Calculate Tax"):
    slabs = tax_slabs[year]
    tax_table_data = calculate_tax_table(taxable_income, slabs)
    
    # Convert tax table data to DataFrame for display
    df = pd.DataFrame(tax_table_data)
    
    # Display the tax table
    st.subheader(f"Tax Calculation Table for {year}")
    st.dataframe(df)

    # Show final tax liability
    if tax_table_data:
        final_row = tax_table_data[-1]
        st.success(f"Your total tax liability for {year} is: {final_row['Final Tax After Surcharge and Cess (₹)']}")
