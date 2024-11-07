def highlight_total_row(row):
    if row.iloc[0] == 'Total':
        return ['font-weight: bold; background-color: #F0F2F6' for _ in row]
    else:
        return ['' for _ in row]
