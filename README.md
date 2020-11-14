# Loans analysis & prediction

Loans database analysis. 
Analysis of doubtful and lost debts from the business loans portfolio. 
Statistical analysis by demographic, business, and sectoral characteristics. 
Then, using machine learning models, prediction of loan repayment prospects according to different parameters.

File: loans_project.ipynb
This file includes an analysis of the data and building the prediction model.

File: Loan_RF_Classifier.pckl
This file is the prediction model to be imported and used in the main file and the main function to run for predictions.

File: loans_df.py
The file includes a unified dataframe of readable categories values in columns, 
and the corresponding coded categories columns as integer values.
The coded values are used in the prediction model.

File: loanDictionary.py
Create a dictionary of the loans_df dataframe with categories names as keys, and coded categories as integer values
The dictionary will be used to get user input of the different categories (keys)
and pass the corresponding coded value of this category to the prediction model.

File: ClassApplication.py
Defines loan application information/ details.
The prediction model will give a prediction results based on loan application information.

File: loanPredict.py
Main file, main function to run for the prediction model

