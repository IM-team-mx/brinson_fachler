import pandas as pd
import numpy as np
import streamlit as st
import datetime

debug = False


def load_data():
    classif_file = './Input files/equities_classifications.csv'
    if debug:
        ptf_file = './Input files/portfolios.csv'
        bm_file = './Input files/benchmarks.csv'
    else:
        ptf_file = col1.file_uploader('portfolios.csv file', help='File produced by the Performance service '
                                                                  'via PerfContribution.sh --action export')
        bm_file = col2.file_uploader('benchmarks.csv file', help='File produced by the Performance service '
                                                                 'via PerfContribution.sh --action export')

    return ptf_file, bm_file, classif_file


def prepare_data(ptf_df, bm_df, classifications_df, classification):
    # Merge portfolio and benchmark data with classifications data on Instrument/Product
    ptf_df = ptf_df.merge(classifications_df, left_on='Instrument', right_on='Product', how='left')
    ptf_df = ptf_df.fillna('Cash')
    bm_df = bm_df.merge(classifications_df, left_on='Instrument', right_on='Product', how='left')

    # Filter by relevant columns for analysis
    portfolio_columns = ['Instrument', 'Start Date', 'End Date', 'DeltaMv', 'PreviousMv', 'Product description',
                         classification]
    benchmark_columns = ['Instrument', 'Start Date', 'End Date', 'DeltaMv', 'PreviousMv', 'Product description',
                         classification]
    ptf_df = ptf_df[portfolio_columns]
    bm_df = bm_df[benchmark_columns]

    # Full outer merge to include all rows from both portfolio_df and benchmark_df
    merged_df = pd.merge(
        ptf_df,
        bm_df,
        on=['Instrument', 'Product description', 'Start Date', 'End Date', classification],
        how='outer',
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

    bf_df['Total_Return_benchmark'] = \
        bf_df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') / \
        bf_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

    bf_df['Total_Return_portfolio'] = \
        bf_df.groupby('Start Date')['DeltaMv_portfolio'].transform('sum') / \
        bf_df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

    # Apply the Brinson Fachler formula (version where interaction effect is counted as Selection)
    bf_df['Allocation Effect'] = (bf_df['Weight_portfolio'] - bf_df['Weight_benchmark']) * \
                                 (bf_df['Return_benchmark'] - bf_df['Total_Return_benchmark'])

    bf_df['Selection Effect'] = bf_df['Weight_portfolio'] * (bf_df['Return_portfolio'] - bf_df['Return_benchmark'])

    return bf_df


def brinson_fachler_instrument(prepared_data_df, classif_criteria, classif_value):
    bf_df = prepared_data_df

    # Calculate weights and returns
    bf_df['Weight_portfolio'] = bf_df['PreviousMv_portfolio'] / bf_df.groupby('Start Date')[
        'PreviousMv_portfolio'].transform('sum')
    bf_df['Weight_benchmark'] = bf_df['PreviousMv_benchmark'] / bf_df.groupby('Start Date')[
        'PreviousMv_benchmark'].transform('sum')

    # Remove rows for which both the portfolio and benchmark weights are 0
    bf_df = bf_df[(bf_df['Weight_portfolio'] != 0) | (bf_df['Weight_benchmark'] != 0)]  # to be changed

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
        bf_df[classif_criteria] == classif_value
        ]

    # Add Total Level benchmark return column
    bf_df['Total_Level_Return_benchmark'] = \
        bf_df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') / \
        bf_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

    bf_df['Selection Effect'] = \
        bf_df['Weight_portfolio'] * (bf_df['Return_portfolio'] - bf_df['Total_Level_Return_benchmark'])

    return bf_df


def brinson_fachler_level_two(prepared_data_df, classif_criteria, classif_value):
    bf_df = prepared_data_df

    # Calculate weights and returns
    bf_df['Weight_portfolio'] = bf_df['PreviousMv_portfolio'] / bf_df.groupby('Start Date')[
        'PreviousMv_portfolio'].transform('sum')
    bf_df['Weight_benchmark'] = bf_df['PreviousMv_benchmark'] / bf_df.groupby('Start Date')[
        'PreviousMv_benchmark'].transform('sum')

    # Remove rows for which both the portfolio and benchmark weights are 0
    bf_df = bf_df[(bf_df['Weight_portfolio'] != 0) | (bf_df['Weight_benchmark'] != 0)]  # to be changed

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
        bf_df[classif_criteria] == classif_value
        ]

    # Add Total Level weights columns
    bf_df['Total_Level_Weight_benchmark'] = bf_df.groupby('Start Date')['Weight_benchmark'].transform('sum')
    bf_df['Total_Level_Weight_portfolio'] = bf_df.groupby('Start Date')['Weight_portfolio'].transform('sum')

    # Add Total Level benchmark return column
    bf_df['Total_Level_Return_benchmark'] = \
        bf_df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') / \
        bf_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')
    bf_df['Total_Level_Return_portfolio'] = \
        bf_df.groupby('Start Date')['DeltaMv_portfolio'].transform('sum') / \
        bf_df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

    bf_df['Allocation Effect'] = (
                                         bf_df['Weight_portfolio']
                                         - bf_df['Weight_benchmark']
                                         * bf_df['Total_Level_Weight_portfolio']
                                         / bf_df['Total_Level_Weight_benchmark']
                                 ) * (bf_df['Return_benchmark'] - bf_df['Total_Level_Return_benchmark'])

    bf_df['Selection Effect'] = bf_df['Weight_portfolio'] * (bf_df['Return_portfolio'] - bf_df['Return_benchmark'])

    return bf_df


def total_returns(attribution_df, start_date):
    start_date = pd.to_datetime(start_date)
    attribution_df['Start Date'] = pd.to_datetime(attribution_df['Start Date'])
    attribution_df = attribution_df[attribution_df['Start Date'] >= start_date]

    ptf_bm_returns_df = attribution_df.groupby('Start Date').min()[['Total_Return_portfolio',
                                                                    'Total_Return_benchmark']].reset_index()
    ptf_bm_returns_df['GRAP factor'] = np.nan

    # Calculate GRAP factor for each row
    for i in range(len(ptf_bm_returns_df)):
        # Product of Portfolio returns for dates strictly smaller than the current row date
        portfolio_product = (1 + ptf_bm_returns_df.loc[:i - 1, 'Total_Return_portfolio']).prod() if i > 0 else 1

        # Product of Benchmark returns for dates strictly larger than the current row date
        benchmark_product = (1 + ptf_bm_returns_df.loc[i + 1:, 'Total_Return_benchmark']).prod()

        # Calculate GRAP factor as the product of the two
        ptf_bm_returns_df.loc[i, 'GRAP factor'] = portfolio_product * benchmark_product

    return ptf_bm_returns_df


def grap_smoothing(attribution_df, ptf_bm_returns_df, classif_criteria):
    attribution_df['Start Date'] = pd.to_datetime(attribution_df['Start Date'])
    attribution_df = attribution_df.merge(ptf_bm_returns_df[['Start Date', 'GRAP factor']], on='Start Date', how='left')

    # Calculating Smoothed Allocation and Smoothed Selection
    attribution_df['Selection'] = attribution_df['Selection Effect'] * attribution_df['GRAP factor']
    if classif_criteria != '':
        attribution_df['Allocation'] = attribution_df['Allocation Effect'] * attribution_df['GRAP factor']
        attribution_df['Excess return'] = attribution_df['Allocation'] + attribution_df['Selection']
        grap_result_df = attribution_df.groupby(classif_criteria)[
            ['Excess return', 'Allocation', 'Selection']].sum().reset_index()
    else:
        # attribution_df['Allocation + Selection'] = attribution_df['Allocation'] + attribution_df['Selection']
        grap_result_df = attribution_df.groupby(['Product description', 'Instrument'])[
            ['Selection']].sum().reset_index()

    # Summing the Smoothed Allocation and Smoothed Selection by Sector across dates
    grap_result_df.loc['Total'] = grap_result_df.sum()
    if classif_criteria != '':
        grap_result_df.loc[grap_result_df.index[-1], classif_criteria] = 'Total'
    else:
        # grap_result_df.loc[grap_result_df.index[-1], 'Instrument'] = 'Total'
        grap_result_df.loc[grap_result_df.index[-1], ['Product description', 'Instrument']] = 'Total'

    return grap_result_df


def highlight_total_row(row):
    if row.iloc[0] == 'Total' == 'Total':
        return ['font-weight: bold; background-color: #F0F2F6' for _ in row]
    else:
        return ['' for _ in row]


st.set_page_config(
    page_title='Brinson Fachler analysis',
    page_icon=':bar_chart:',
    layout='wide'
)
st.markdown('### :bar_chart: Brinson-Fachler performance attribution')

col1, col2 = st.columns([0.5, 0.5])
col3, col4 = st.columns([0.3, 0.7])
col5, col6 = st.columns([0.3, 0.7])

reference_date = col1.date_input('Start date', datetime.date(2019, 12, 31))
decimal_places = col2.selectbox(
    'Decimal places',
    (2, 4, 8, 12),
)

portfolios_file, benchmarks_file, classifications_file = load_data()

if portfolios_file is not None and benchmarks_file is not None:
    # Load the data files, replacing NaNs with zeros
    portfolio_df = pd.read_csv(portfolios_file).fillna(0)
    benchmark_df = pd.read_csv(benchmarks_file).fillna(0)
    equities_classifications_df = pd.read_csv(classifications_file).fillna(0)

    # Execute Brinson-Fachler Analysis with the selected criteria
    classification_criteria = col3.radio(
        'Allocation criteria',
        ['GICS sector', 'GICS industry group', 'GICS industry', 'GICS sub-industry', 'Region', 'Country'],
    )
    data = prepare_data(portfolio_df, benchmark_df, equities_classifications_df, classification_criteria)
    brinson_fachler_result = brinson_fachler_analysis(data, classification_criteria)

    total_returns_df = total_returns(brinson_fachler_result, reference_date)
    grap_result = grap_smoothing(brinson_fachler_result, total_returns_df, classification_criteria)

    df_style = '{:,.' + str(decimal_places) + '%}'

    styled_grap_result_df = grap_result.style.apply(highlight_total_row, axis=1)
    styled_grap_result_df = styled_grap_result_df.format({
        'Allocation': df_style.format,
        'Selection': df_style.format,
        'Excess return': df_style.format
    })

    col4.markdown('**Brinson-Fachler attribution**:', help=f'For this model we consider that if the benchmark is not '
                                                           f'invested in a {classification_criteria}, the selection '
                                                           f'effect is 0 and the excess return is equal to the '
                                                           f'allocation effect')
    col4.dataframe(styled_grap_result_df, hide_index=True, width=700, height=(len(grap_result.index) + 1) * 35 + 3)

    classification_values = grap_result[classification_criteria].to_list()
    classification_values = [item for item in classification_values if item not in ['Cash', 'Total']]
    classification_value = col5.radio(f'Select a {classification_criteria}:', classification_values)

    brinson_fachler_instrument_result = brinson_fachler_instrument(data, classification_criteria, classification_value)
    grap_instrument_result = grap_smoothing(brinson_fachler_instrument_result, total_returns_df, '')

    styled_grap_instrument_result_df = grap_instrument_result.style.apply(highlight_total_row, axis=1)
    styled_grap_instrument_result_df = styled_grap_instrument_result_df.format({
        'Selection': df_style.format,
    })
    col6.markdown('**Instrument selection details**:', help='The Selection total value should match the Selection '
                                                            'value in the main view')
    col6.dataframe(styled_grap_instrument_result_df, hide_index=True, width=700,
                   height=(len(grap_instrument_result.index) + 1) * 35 + 3)
