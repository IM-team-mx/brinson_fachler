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
