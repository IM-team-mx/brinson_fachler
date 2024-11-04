import pandas as pd
import numpy as np
import streamlit as st
import datetime

debug = False

def load_data():
    classifications_file = './Input files/equities_classifications.csv'
    if debug:
        portfolios_file = './Input files/portfolios.csv'
        benchmarks_file ='./Input files/benchmarks.csv'
    else:
        portfolios_file = col1.file_uploader("portfolios.csv file")
        benchmarks_file = col2.file_uploader("benchmarks.csv file")

    return portfolios_file, benchmarks_file, classifications_file


def prepare_data(ptf_df, bm_df, classifications_df, classification):
    # Merge portfolio and benchmark data with classifications data on Instrument/Product
    ptf_df = ptf_df.merge(classifications_df, left_on="Instrument", right_on="Product", how="left")
    ptf_df = ptf_df.fillna("Cash")
    bm_df = bm_df.merge(classifications_df, left_on="Instrument", right_on="Product", how="left")

    # Filter by relevant columns for analysis
    portfolio_columns = ["Instrument", "Start Date", "End Date", "DeltaMv", "PreviousMv", classification]
    benchmark_columns = ["Instrument", "Start Date", "End Date", "DeltaMv", "PreviousMv", classification]
    ptf_df = ptf_df[portfolio_columns]
    bm_df = bm_df[benchmark_columns]

    # Full outer merge to include all rows from both portfolio_df and benchmark_df
    merged_df = pd.merge(
        ptf_df,
        bm_df,
        on=["Instrument", "Start Date", "End Date", classification],
        how="outer",
        suffixes=('_portfolio', '_benchmark')
    ).fillna(0)  # Fill NaNs with zeros for unmatched data

    return merged_df


def brinson_fachler_analysis(prepared_data_df, classification):

    bf_df = prepared_data_df.groupby(['Start Date', classification]).agg({
        'DeltaMv_portfolio': 'sum',
        'DeltaMv_benchmark': 'sum',
        'PreviousMv_portfolio': 'sum',
        'PreviousMv_benchmark': 'sum'
    }).reset_index()

    # Calculate weights and returns
    bf_df['Weight_portfolio'] = bf_df['PreviousMv_portfolio'] / bf_df.groupby('Start Date')[
        'PreviousMv_portfolio'].transform('sum')
    bf_df['Weight_benchmark'] = bf_df['PreviousMv_benchmark'] / bf_df.groupby('Start Date')[
        'PreviousMv_benchmark'].transform('sum')

    # Remove rows for which both the portfolio and benchmark weights are 0
    bf_df = bf_df[(bf_df['Weight_portfolio'] != 0) | (bf_df['Weight_benchmark'] != 0)]

    bf_df['Return_portfolio'] = bf_df.apply(
        lambda row: row['DeltaMv_portfolio'] / row['PreviousMv_portfolio']
        if row['Weight_portfolio'] != 0
        else row['DeltaMv_benchmark'] / row['PreviousMv_benchmark'],
        axis=1
    )
    bf_df['Return_benchmark'] = bf_df.apply(
        lambda row: row['DeltaMv_benchmark'] / row['PreviousMv_benchmark']
        if row['Weight_benchmark'] != 0
        else row['DeltaMv_portfolio'] / row['PreviousMv_portfolio'],
        axis=1
    )

    bf_df['Total_Return_benchmark'] = bf_df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') / \
                                      bf_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

    bf_df['Total_Return_portfolio'] = bf_df.groupby('Start Date')['DeltaMv_portfolio'].transform('sum') / \
                                      bf_df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

    # Apply the Brinson Fachler formula (version where interaction effect is counted as Selection)
    bf_df['Allocation Effect'] = (bf_df['Weight_portfolio'] - bf_df['Weight_benchmark']) * \
                              (bf_df['Return_benchmark'] - bf_df[
                                  'Total_Return_benchmark'])

    bf_df['Selection Effect'] = bf_df['Weight_portfolio'] * (
            bf_df['Return_portfolio'] - bf_df['Return_benchmark'])

    return bf_df


def brinson_fachler_instrument(prepared_data_df, classification_criteria, classification_value):
    bf_df = prepared_data_df

    # Calculate weights and returns
    bf_df['Weight_portfolio'] = bf_df['PreviousMv_portfolio'] / bf_df.groupby('Start Date')[
    'PreviousMv_portfolio'].transform('sum')
    bf_df['Weight_benchmark'] = bf_df['PreviousMv_benchmark'] / bf_df.groupby('Start Date')[
    'PreviousMv_benchmark'].transform('sum')

    # Remove rows for which both the portfolio and benchmark weights are 0
    bf_df = bf_df[(bf_df['Weight_portfolio'] != 0) | (bf_df['Weight_benchmark'] != 0)]

    bf_df['Return_portfolio'] = bf_df.apply(
        lambda row: row['DeltaMv_portfolio'] / row['PreviousMv_portfolio']
        if row['Weight_portfolio'] != 0
        else row['DeltaMv_benchmark'] / row['PreviousMv_benchmark'],
        axis=1
    )
    bf_df['Return_benchmark'] = bf_df.apply(
        lambda row: row['DeltaMv_benchmark'] / row['PreviousMv_benchmark']
        if row['Weight_benchmark'] != 0
        else row['DeltaMv_portfolio'] / row['PreviousMv_portfolio'],
        axis=1
    )

    # Filter the dataframe on the matching classification
    bf_df = bf_df[
        bf_df[classification_criteria] == classification_value
    ]

    # Add Total Level weights columns
    bf_df['Total_Level_Weight_benchmark'] = bf_df.groupby('Start Date')['Weight_benchmark'].transform('sum')
    bf_df['Total_Level_Weight_portfolio'] = bf_df.groupby('Start Date')['Weight_portfolio'].transform('sum')

    # Add Total Level benchmark return column
    bf_df['Total_Level_Return_benchmark'] = bf_df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') / \
                                         bf_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')
    bf_df['Total_Level_Return_portfolio'] = bf_df.groupby('Start Date')['DeltaMv_portfolio'].transform('sum') / \
                                         bf_df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

    bf_df['Allocation Effect'] = (bf_df['Weight_portfolio'] - bf_df['Weight_benchmark'] * bf_df['Total_Level_Weight_portfolio'] /
                               bf_df['Total_Level_Weight_benchmark']) * (
                                      bf_df['Return_benchmark'] - bf_df['Total_Level_Return_benchmark'])

    bf_df['Selection Effect'] = bf_df['Weight_portfolio'] * (bf_df['Return_portfolio'] - bf_df['Return_benchmark'])

    return bf_df


def total_returns(attribution_df, start_date):
    start_date = pd.to_datetime(start_date)
    attribution_df['Start Date'] = pd.to_datetime(attribution_df['Start Date'])
    attribution_df = attribution_df[attribution_df['Start Date'] >= start_date]

    total_returns_df = attribution_df.groupby('Start Date').min()[['Total_Return_portfolio',
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

    return total_returns_df


def grap_smoothing(attribution_df, total_returns_df, classification_criteria):
    attribution_df['Start Date'] = pd.to_datetime(attribution_df['Start Date'])
    attribution_df = attribution_df.merge(total_returns_df[['Start Date', 'GRAP factor']], on='Start Date', how='left')

    # Calculating Smoothed Allocation and Smoothed Selection
    attribution_df['Allocation'] = attribution_df['Allocation Effect'] * attribution_df['GRAP factor']
    attribution_df['Selection'] = attribution_df['Selection Effect'] * attribution_df['GRAP factor']
    if classification_criteria != "":
        attribution_df['Excess return'] = attribution_df['Allocation'] + attribution_df['Selection']
        grap_result_df = attribution_df.groupby(classification_criteria)[
            ["Excess return", "Allocation", "Selection"]].sum().reset_index()
    else:
        attribution_df['Allocation + Selection'] = attribution_df['Allocation'] + attribution_df['Selection']
        grap_result_df = attribution_df.groupby('Instrument')[
            ["Allocation", "Selection", "Allocation + Selection"]].sum().reset_index()

    # Summing the Smoothed Allocation and Smoothed Selection by Sector across dates
    grap_result_df.loc['Total'] = grap_result_df.sum()
    if classification_criteria != "":
        grap_result_df.loc[grap_result_df.index[-1], classification_criteria] = 'Total'
    else:
        grap_result_df.loc[grap_result_df.index[-1], 'Instrument'] = 'Total'

    return grap_result_df


st.set_page_config(
    page_title="Brinson Fachler analysis",
    page_icon=":bar_chart:",
    layout="wide"
)
st.markdown("### :bar_chart: Brinson-Fachler performance attribution")

col1, col2 = st.columns([0.5, 0.5])
col3, col4 = st.columns([0.3, 0.7])
col5, col6 = st.columns([0.3, 0.7])

reference_date = col1.date_input("Start date", datetime.date(2019, 12, 31))
decimal_places = col2.selectbox(
        "Decimal places",
        (2, 4, 8, 12),
    )

portfolios_file, benchmarks_file, classifications_file = load_data()

if portfolios_file is not None and benchmarks_file is not None:
    classification_criteria = col3.radio(
        "Allocation criteria",
        ["GICS sector", "GICS industry group", "GICS industry", "GICS sub-industry", "Region", "Country"],
    )

    # Load the data files, replacing NaNs with zeros
    portfolio_df = pd.read_csv(portfolios_file).fillna(0)
    benchmark_df = pd.read_csv(benchmarks_file).fillna(0)
    equities_classifications_df = pd.read_csv(classifications_file).fillna(0)

    # Execute Brinson-Fachler Analysis with the selected criteria
    data = prepare_data(portfolio_df, benchmark_df, equities_classifications_df, classification_criteria)
    brinson_fachler_result = brinson_fachler_analysis(data, classification_criteria)

    total_returns_df = total_returns(brinson_fachler_result, reference_date)
    grap_result = grap_smoothing(brinson_fachler_result, total_returns_df, classification_criteria)

    df_style = '{:,.' + str(decimal_places) + '%}'

    grap_result_display = grap_result.style.format({
        'Allocation': df_style.format,
        'Selection': df_style.format,
        'Excess return': df_style.format
    })
    col4.markdown("**Brinson-Fachler attribution**:", help="Allocation + Selection total value should match the Selection"
                                                  " value in the main view")
    col4.dataframe(grap_result_display, hide_index=True, width=700, height=(len(grap_result.index) + 1) * 35 + 3)

    classification_values = grap_result[classification_criteria].to_list()
    classification_values = [item for item in classification_values if item not in ["Cash", "Total"]]
    classification_value = col5.radio(f"Select a {classification_criteria}:", classification_values)

    brinson_fachler_instrument_result = brinson_fachler_instrument(data, classification_criteria, classification_value)

    grap_instrument_result = grap_smoothing(brinson_fachler_instrument_result, total_returns_df, "")

    grap_instrument_result_display = grap_instrument_result.style.format({
        'Allocation': df_style.format,
        'Selection': df_style.format,
        'Allocation + Selection': df_style.format
    })
    col6.markdown("**Instrument details**:", help="Allocation + Selection total value should match the Selection"
                                                " value in the main view")
    col6.dataframe(grap_instrument_result_display, hide_index=True, width=700, height=(len(grap_instrument_result.index) + 1) * 35 + 3)
