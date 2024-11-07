import pandas as pd

def prepare_data(ptf_df, bm_df, classifications_df, classification):
    # Merges and cleans data as needed
    ptf_df = ptf_df.merge(classifications_df, left_on='Instrument', right_on='Product', how='left').fillna('Cash')
    bm_df = bm_df.merge(classifications_df, left_on='Instrument', right_on='Product', how='left')

    portfolio_columns = ['Instrument', 'Start Date', 'End Date', 'DeltaMv', 'PreviousMv', 'Product description', classification]
    benchmark_columns = ['Instrument', 'Start Date', 'End Date', 'DeltaMv', 'PreviousMv', 'Product description', classification]
    ptf_df = ptf_df[portfolio_columns]
    bm_df = bm_df[benchmark_columns]

    merged_df = pd.merge(ptf_df, bm_df, on=['Instrument', 'Product description', 'Start Date', 'End Date', classification], how='outer', suffixes=('_portfolio', '_benchmark')).fillna(0)

    return merged_df
