def brinson_fachler(prepared_data_df, classification):
    bf_df = prepared_data_df.groupby(['Start Date', classification]).agg({
        'DeltaMv_portfolio': 'sum',
        'DeltaMv_benchmark': 'sum',
        'PreviousMv_portfolio': 'sum',
        'PreviousMv_benchmark': 'sum'
    }).reset_index()

    # Calculate weights and returns
    bf_df['Weight_portfolio'] = \
        bf_df['PreviousMv_portfolio'] / bf_df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

    bf_df['Weight_benchmark'] = \
        bf_df['PreviousMv_benchmark'] / bf_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

    bf_df['Return_portfolio'] = bf_df.apply(
        lambda row: row['DeltaMv_portfolio'] / row['PreviousMv_portfolio']
        if row['Weight_portfolio'] != 0
        else row['DeltaMv_benchmark'] / row['PreviousMv_benchmark'], axis=1
    )

    bf_df['Return_benchmark'] = bf_df.apply(
        lambda row: row['DeltaMv_benchmark'] / row['PreviousMv_benchmark']
        if row['Weight_benchmark'] != 0
        else row['DeltaMv_portfolio'] / row['PreviousMv_portfolio'], axis=1)

    bf_df['Total_Return_benchmark'] = \
        bf_df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') \
        / bf_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

    bf_df['Total_Return_portfolio'] = \
        bf_df.groupby('Start Date')['DeltaMv_portfolio'].transform('sum')\
        / bf_df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

    bf_df['Allocation Effect'] = \
        (bf_df['Weight_portfolio'] - bf_df['Weight_benchmark']) \
        * (bf_df['Return_benchmark'] - bf_df['Total_Return_benchmark'])

    bf_df['Selection Effect'] = bf_df['Weight_portfolio'] * (bf_df['Return_portfolio'] - bf_df['Return_benchmark'])

    return bf_df
