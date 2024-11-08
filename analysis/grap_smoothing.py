import pandas as pd


def grap_smoothing(attribution_df, ptf_bm_returns_df, model, classif_criteria):
    attribution_df['Start Date'] = pd.to_datetime(attribution_df['Start Date'])
    attribution_df = attribution_df.merge(ptf_bm_returns_df[['Start Date', 'GRAP factor']], on='Start Date', how='left')

    # GRAP factor application
    attribution_df['Selection'] = attribution_df['Selection Effect'] * attribution_df['GRAP factor']

    if classif_criteria != '':
        attribution_df['Allocation'] = attribution_df['Allocation Effect'] * attribution_df['GRAP factor']
        if model == 'Brinson-Hood-Beebower':
            attribution_df['Interaction'] = attribution_df['Interaction Effect'] * attribution_df['GRAP factor']
            attribution_df['Excess return'] = attribution_df['Allocation'] \
                                              + attribution_df['Interaction'] + attribution_df['Selection']
            grap_result_df = attribution_df.groupby(classif_criteria)[
                ['Excess return', 'Allocation', 'Selection', 'Interaction']].sum().reset_index()
        else:
            attribution_df['Excess return'] = attribution_df['Allocation'] + attribution_df['Selection']
            grap_result_df = attribution_df.groupby(classif_criteria)[
                ['Excess return', 'Allocation', 'Selection']].sum().reset_index()

    else:
        grap_result_df = attribution_df.groupby(['Product description', 'Instrument'])[['Selection']].sum().reset_index()

    grap_result_df.loc['Total'] = grap_result_df.sum()
    if classif_criteria != '':
        grap_result_df.loc[grap_result_df.index[-1], classif_criteria] = 'Total'
    else:
        grap_result_df.loc[grap_result_df.index[-1], ['Product description', 'Instrument']] = 'Total'

    return grap_result_df
