import pandas as pd
import numpy as np
import datetime
import streamlit as st

# Load the data files, replacing NaNs with zeros
portfolio_df = pd.read_csv('./Input files/portfolios.csv').fillna(0)
benchmark_df = pd.read_csv('./Input files/benchmarks.csv').fillna(0)
equities_classifications_df = pd.read_csv('Input files/equities_classifications.csv').fillna(0)
reference_date = "2019-12-31"


classification_criteria = "GICS sector"
classification_value = "Communication Services"

# Merge portfolio and benchmark data with classifications data on Instrument/Product
ptf_df = portfolio_df.merge(equities_classifications_df, left_on="Instrument", right_on="Product", how="left")
ptf_df = ptf_df.fillna("Cash")
bm_df = benchmark_df.merge(equities_classifications_df, left_on="Instrument", right_on="Product", how="left")

# Filter by relevant columns for analysis
portfolio_columns = ["Instrument", "Start Date", "End Date", "DeltaMv", "PreviousMv", classification_criteria]
benchmark_columns = ["Instrument", "Start Date", "End Date", "DeltaMv", "PreviousMv", classification_criteria]
ptf_df = ptf_df[portfolio_columns]
bm_df = bm_df[benchmark_columns]

# Full outer merge to include all rows from both portfolio_df and benchmark_df
df = pd.merge(
    ptf_df,
    bm_df,
    on=["Instrument", "Start Date", "End Date", classification_criteria],
    how="outer",
    suffixes=('_portfolio', '_benchmark')
).fillna(0)  # Fill NaNs with zeros for unmatched data

# Calculate weights and contributions
df['Weight_portfolio'] = df['PreviousMv_portfolio'] / df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')
df['Weight_benchmark'] = df['PreviousMv_benchmark'] / df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

# Calculate Portfolio and Benchmark total returns
df['Total_Return_portfolio'] = df.groupby('Start Date')['DeltaMv_portfolio'].transform('sum') / \
                               df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')
df['Total_Return_benchmark'] = df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') / \
                               df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

# Filter data on the classification criteria and classification value selected
df = df[df[classification_criteria] == classification_value]

# Remove rows for which both the portfolio and benchmark weights are 0
df = df[(df['Weight_portfolio'] != 0) | (df['Weight_benchmark'] != 0)]

df['Return_portfolio'] = df.apply(
    lambda row: row['DeltaMv_portfolio'] / row['PreviousMv_portfolio']
    if row['Weight_portfolio'] != 0
    else row['DeltaMv_benchmark'] / row['PreviousMv_benchmark'],
    axis=1
)
df['Return_benchmark'] = df.apply(
    lambda row: row['DeltaMv_benchmark'] / row['PreviousMv_benchmark']
    if row['Weight_benchmark'] != 0
    else row['DeltaMv_portfolio'] / row['PreviousMv_portfolio'],
    axis=1
)

# Add Total Level weights columns
df['Total_Level_Weight_benchmark'] = df.groupby('Start Date')['Weight_benchmark'].transform('sum')
df['Total_Level_Weight_portfolio'] = df.groupby('Start Date')['Weight_portfolio'].transform('sum')

# Add Total Level benchmark return column
df['Total_Level_Return_benchmark'] = df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') / \
                               df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')
df['Total_Level_Return_portfolio'] = df.groupby('Start Date')['DeltaMv_portfolio'].transform('sum') / \
                               df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

df['Allocation Effect'] = (df['Weight_portfolio'] - df['Weight_benchmark']) *\
                          (df['Total_Level_Return_benchmark'] - df['Total_Return_benchmark'])
df['Selection Effect'] = df['Weight_portfolio'] * \
                         (df['Total_Level_Return_portfolio'] - df['Total_Level_Return_benchmark'])

# Convert date columns to datetime
df['Start Date'] = pd.to_datetime(df['Start Date'])
df['End Date'] = pd.to_datetime(df['End Date'])

# reference_date = st.date_input("Start date", datetime.date(2019, 12, 31))
reference_date = pd.to_datetime(reference_date)

df = df[df['Start Date'] >= reference_date]

total_returns_df = df.groupby('Start Date').min()[['Total_Return_portfolio',
                                                   'Total_Return_benchmark']].reset_index()
total_returns_df['GRAP factor'] = np.nan

# Calculate GRAP factor for each row
for i in range(len(total_returns_df)):
    # Product of Portfolio returns for dates strictly smaller than the current row date
    portfolio_product = (1 + total_returns_df.loc[:i - 1, 'Total_Return_portfolio']).prod() if i > 0 else 1

    # Product of Benchmark returns for dates strictly larger than the current row date
    benchmark_product = (1 + total_returns_df.loc[i + 1:, 'Total_Return_benchmark']).prod()

    # Calculate GRAP factor as the product of the two
    total_returns_df.loc[i, 'GRAP factor'] = portfolio_product * benchmark_product

# Merging the GRAP factor from total_returns_df into df based on the date
df = df.merge(total_returns_df[['Start Date', 'GRAP factor']], on='Start Date', how='left')

# Calculating Smoothed Allocation and Smoothed Selection
df['Allocation'] = df['Allocation Effect'] * df['GRAP factor']
df['Selection'] = df['Selection Effect'] * df['GRAP factor']
df['Excess return'] = df['Allocation'] + df['Selection']

# Summing the Smoothed Allocation and Smoothed Selection by Sector across dates
grap_result = df.groupby('Instrument')[["Allocation", "Selection", "Excess return"]].sum().reset_index()

decimal_places = st.selectbox(
        "Decimal places",
        (2, 4, 8, 12),
    )

grap_result.loc['Total']= grap_result.sum()
grap_result.loc[grap_result.index[-1], 'Instrument'] = 'Total'

df_style = '{:,.' + str(decimal_places) + '%}'

grap_result = grap_result.style.format({
    'Allocation': df_style.format,
    'Selection': df_style.format,
    'Excess return': df_style.format
})

st.dataframe(grap_result, hide_index=True, width=700, height=(len(grap_result.index) + 1) * 35 + 3)