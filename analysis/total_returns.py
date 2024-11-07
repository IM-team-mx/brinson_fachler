import pandas as pd
import numpy as np

def total_returns(attribution_df, start_date):
    start_date = pd.to_datetime(start_date)
    attribution_df['Start Date'] = pd.to_datetime(attribution_df['Start Date'])
    attribution_df = attribution_df[attribution_df['Start Date'] >= start_date]

    ptf_bm_returns_df = attribution_df.groupby('Start Date').min()[['Total_Return_portfolio', 'Total_Return_benchmark']].reset_index()
    ptf_bm_returns_df['GRAP factor'] = np.nan

    for i in range(len(ptf_bm_returns_df)):
        portfolio_product = (1 + ptf_bm_returns_df.loc[:i - 1, 'Total_Return_portfolio']).prod() if i > 0 else 1
        benchmark_product = (1 + ptf_bm_returns_df.loc[i + 1:, 'Total_Return_benchmark']).prod()
        ptf_bm_returns_df.loc[i, 'GRAP factor'] = portfolio_product * benchmark_product

    return ptf_bm_returns_df
