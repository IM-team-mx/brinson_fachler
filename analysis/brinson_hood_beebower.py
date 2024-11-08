def brinson_hood_beebower(prepared_data_df, classification):
    bhb_df = prepared_data_df.groupby(['Start Date', classification]).agg({
        'DeltaMv_portfolio': 'sum',
        'DeltaMv_benchmark': 'sum',
        'PreviousMv_portfolio': 'sum',
        'PreviousMv_benchmark': 'sum'
    }).reset_index()

    # Calculate weights and returns
    bhb_df['Weight_portfolio'] = \
        bhb_df['PreviousMv_portfolio'] / bhb_df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

    bhb_df['Weight_benchmark'] = \
        bhb_df['PreviousMv_benchmark'] / bhb_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

    bhb_df['Return_portfolio'] = bhb_df.apply(
        lambda row: row['DeltaMv_portfolio'] / row['PreviousMv_portfolio']
        if row['Weight_portfolio'] != 0
        else row['DeltaMv_benchmark'] / row['PreviousMv_benchmark'], axis=1
    )

    bhb_df['Return_benchmark'] = bhb_df.apply(
        lambda row: row['DeltaMv_benchmark'] / row['PreviousMv_benchmark']
        if row['Weight_benchmark'] != 0
        else row['DeltaMv_portfolio'] / row['PreviousMv_portfolio'], axis=1)

    bhb_df['Total_Return_benchmark'] = \
        bhb_df.groupby('Start Date')['DeltaMv_benchmark'].transform('sum') \
        / bhb_df.groupby('Start Date')['PreviousMv_benchmark'].transform('sum')

    bhb_df['Total_Return_portfolio'] = \
        bhb_df.groupby('Start Date')['DeltaMv_portfolio'].transform('sum')\
        / bhb_df.groupby('Start Date')['PreviousMv_portfolio'].transform('sum')

    # Compute Allocation, Selection and Interaction effects of the BHB model
    bhb_df['Allocation Effect'] = (bhb_df['Weight_portfolio'] - bhb_df['Weight_benchmark']) * bhb_df['Return_benchmark']

    bhb_df['Selection Effect'] = bhb_df['Weight_benchmark'] * (bhb_df['Return_portfolio'] - bhb_df['Return_benchmark'])

    bhb_df['Interaction Effect'] = \
        (bhb_df['Weight_portfolio'] - bhb_df['Weight_benchmark'])\
        * (bhb_df['Return_portfolio'] - bhb_df['Return_benchmark'])

    return bhb_df
