import pandas as pd

loans_df = pd.read_pickle('loans_df')

# Create a dictionary of the dataframe string values as keys and the coded categories valuse as int
keys = [(loans_df.columns[k]) for k in range(9)]
vals = [(loans_df.columns[v]) for v in range(9,18)]

loan_dict = [dict(zip(loans_df[keys[i]],loans_df[vals[i]])) for i in range(9)]
