import pandas as pd

# loans_df includes coded categories values as an integer for the prediction model
loans_df = pd.read_pickle('loans_df')

# Create a dictionary of the loans_df dataframe with categories names as keys, and coded categories as integer values
# The dictionary will be used to get user input of the different categories (keys)
# and pass the corresponding coded value of this category to the prediction model
keys = [(loans_df.columns[k]) for k in range(9)] # Readable string categories as keys
vals = [(loans_df.columns[v]) for v in range(9,18)] # Coded categories as integer values

loan_dict = [dict(zip(loans_df[keys[i]],loans_df[vals[i]])) for i in range(9)]
