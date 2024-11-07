import streamlit as st
import datetime
import pandas as pd
from data.load_data import load_data
from analysis.data_preparation import prepare_data
from analysis.brinson_fachler import brinson_fachler_analysis
from analysis.brinson_fachler_instrument import brinson_fachler_instrument
from analysis.total_returns import total_returns
from analysis.smoothing import grap_smoothing
from utils.styling import highlight_total_row

# Streamlit page configuration
st.set_page_config(
    page_title='Brinson-Fachler Analysis',
    page_icon=':bar_chart:',
    layout='wide'
)
st.markdown('### :bar_chart: Brinson-Fachler Performance Attribution')

# Layout columns
col1, col2 = st.columns([0.5, 0.5])
col3, col4 = st.columns([0.3, 0.7])
col5, col6 = st.columns([0.3, 0.7])

# User inputs
reference_date = col1.date_input('Start date', datetime.date(2019, 12, 31))
decimal_places = col2.selectbox('Decimal places', (2, 4, 8, 12))

# Load data
portfolios_file, benchmarks_file, classifications_file = load_data(col1, col2)

# Main logic for data processing and visualization
if portfolios_file is not None and benchmarks_file is not None:
    # Load the data files, replacing NaNs with zeros
    portfolio_df = pd.read_csv(portfolios_file).fillna(0)
    benchmark_df = pd.read_csv(benchmarks_file).fillna(0)
    classifications_df = pd.read_csv(classifications_file).fillna(0)

    # Execute Brinson-Fachler Analysis with the selected criteria
    classification_criteria = col3.radio(
        'Allocation criteria',
        ['GICS sector', 'GICS industry group', 'GICS industry', 'GICS sub-industry', 'Region', 'Country'],
    )

    # Prepare the data
    prepared_data = prepare_data(portfolio_df, benchmark_df, classifications_df, classification_criteria)
    brinson_fachler_result = brinson_fachler_analysis(prepared_data, classification_criteria)

    # Calculate total returns and apply GRAP smoothing
    total_returns_df = total_returns(brinson_fachler_result, reference_date)
    grap_result = grap_smoothing(brinson_fachler_result, total_returns_df, classification_criteria)

    # Format the DataFrame for display
    df_style = '{:,.' + str(decimal_places) + '%}'
    styled_grap_result_df = grap_result.style.apply(highlight_total_row, axis=1)
    styled_grap_result_df = styled_grap_result_df.format({
        'Allocation': df_style.format,
        'Selection': df_style.format,
        'Excess return': df_style.format
    })

    # Display main analysis results
    col4.markdown('**Brinson-Fachler Attribution**:', help='Details of the Brinson-Fachler performance attribution')
    col4.dataframe(styled_grap_result_df, hide_index=True, width=700, height=(len(grap_result.index) + 1) * 35 + 3)

    # Allow user to drill down by classification
    classification_values = [val for val in grap_result[classification_criteria].to_list() if
                             val not in ['Cash', 'Total']]
    classification_value = col5.radio(f'Select a {classification_criteria}:', classification_values)

    # Drill-down analysis for specific classification

    brinson_fachler_instrument_result = brinson_fachler_instrument(prepared_data, classification_criteria,
                                                                   classification_value)
    grap_instrument_result = grap_smoothing(brinson_fachler_instrument_result, total_returns_df, '')

    # Display detailed instrument-level results
    styled_grap_instrument_result_df = grap_instrument_result.style.apply(highlight_total_row, axis=1)
    styled_grap_instrument_result_df = styled_grap_instrument_result_df.format({'Selection': df_style.format})
    col6.markdown('**Instrument Selection Details**:', help='Detailed selection analysis by instrument')
    col6.dataframe(styled_grap_instrument_result_df, hide_index=True, width=700,
                   height=(len(grap_instrument_result.index) + 1) * 35 + 3)
