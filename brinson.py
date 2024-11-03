import pandas as pd

# Load the data files, replacing NaNs with zeros
portfolio_df = pd.read_csv('Input files/portfolios.csv').fillna(0)
benchmark_df = pd.read_csv('Input files/benchmarks.csv').fillna(0)
equities_classifications_df = pd.read_csv('Input files/equities_classifications.csv').fillna(0)


# Define the Brinson-Fachler Analysis function
def brinson_fachler_analysis(portfolio_df, benchmark_df, classifications_df, classification_criteria):
    """
    Calculate Brinson-Fachler outputs by the specified classification criteria.

    Parameters:
    - portfolio_df: Portfolio data as a DataFrame
    - benchmark_df: Benchmark data as a DataFrame
    - classifications_df: Equities classifications data as a DataFrame
    - classification_criteria: String representing the classification criteria (e.g., "PRODUCT_GICS_SECTOR")

    Returns:
    - Brinson-Fachler summary DataFrames with merged Selection Effect.
    """

    # Merge portfolio and benchmark data with classifications data on Instrument/Product
    portfolio_df = portfolio_df.merge(classifications_df, left_on="Instrument", right_on="Product", how="left")
    portfolio_df = portfolio_df.fillna("Cash")
    benchmark_df = benchmark_df.merge(classifications_df, left_on="Instrument", right_on="Product", how="left")

    # Filter by relevant columns for analysis
    portfolio_columns = ["Instrument", "Start Date", "End Date", "DeltaMv", "PreviousMv", classification_criteria]
    benchmark_columns = ["Instrument", "Start Date", "End Date", "DeltaMv", "PreviousMv", classification_criteria]
    portfolio_df = portfolio_df[portfolio_columns]
    benchmark_df = benchmark_df[benchmark_columns]

    # Full outer merge to include all rows from both portfolio_df and benchmark_df
    merged_df = pd.merge(
        portfolio_df,
        benchmark_df,
        on=["Instrument", "Start Date", "End Date", classification_criteria],
        how="outer",
        suffixes=('_portfolio', '_benchmark')
    ).fillna(0)  # Fill NaNs with zeros for unmatched data

    classif_bf_df = merged_df.groupby(["Start Date", "End Date", classification_criteria]).agg({
        'DeltaMv_portfolio': 'sum',
        'DeltaMv_benchmark': 'sum',
        'PreviousMv_portfolio': 'sum',
        'PreviousMv_benchmark': 'sum'
    }).reset_index()

    # Calculate weights and returns
    classif_bf_df['Weight_portfolio'] = classif_bf_df['PreviousMv_portfolio'] / classif_bf_df.groupby(['Start Date', 'End Date'])[
        'PreviousMv_portfolio'].transform('sum')

    classif_bf_df['Weight_benchmark'] = classif_bf_df['PreviousMv_benchmark'] / classif_bf_df.groupby(['Start Date', 'End Date'])[
        'PreviousMv_benchmark'].transform('sum')

    classif_bf_df['Return_portfolio'] = classif_bf_df['DeltaMv_portfolio'] / classif_bf_df['PreviousMv_portfolio']
    classif_bf_df['Return_benchmark'] = classif_bf_df['DeltaMv_benchmark'] / classif_bf_df['PreviousMv_benchmark']

    classif_bf_df = classif_bf_df.fillna(0)

    # Add a Total Return benchmark column
    classif_bf_df['Total_Return_benchmark'] = classif_bf_df['Weight_benchmark'] * classif_bf_df['Return_benchmark']
    classif_bf_df['Total_Return_benchmark'] = classif_bf_df.groupby('Start Date')['Total_Return_benchmark'].\
        transform('sum')

    # Add a Total Return portfolio column
    classif_bf_df['Total_Return_portfolio'] = classif_bf_df['Weight_portfolio'] * classif_bf_df['Return_portfolio']
    classif_bf_df['Total_Return_portfolio'] = classif_bf_df.groupby('Start Date')['Total_Return_portfolio']. \
        transform('sum')

    # Apply the Brinson Fachler formula (version where interaction effect is counted as Selection)
    classif_bf_df['Allocation Effect'] = (classif_bf_df['Weight_portfolio'] - classif_bf_df['Weight_benchmark']) * \
                                         (classif_bf_df['Return_benchmark'] - classif_bf_df['Total_Return_benchmark'])

    classif_bf_df['Selection Effect'] = classif_bf_df['Weight_portfolio'] * (
                classif_bf_df['Return_portfolio'] - classif_bf_df['Return_benchmark'])

    # Calculate Total Effect
    classif_bf_df['Total Effect'] = classif_bf_df['Allocation Effect'] + classif_bf_df['Selection Effect']

    return classif_bf_df


# Execute Brinson-Fachler Analysis with a sample classification criteria, e.g., "PRODUCT_GICS_SECTOR"
classification_criteria = "Country"

brinson_fachler_result = brinson_fachler_analysis(portfolio_df, benchmark_df, equities_classifications_df,
                                                              classification_criteria)



# Export the summaries to CSV files with specified formatting
brinson_fachler_result.to_csv('./Output files/brinson_fachler_result.csv', sep=';', decimal=',', index=False)
