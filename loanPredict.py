import pandas as pd
import pickle
from loanDictionary import loans_df, loan_dict
from ClassApplication import new_application
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")

# Create a DataFrame with new loan application information
def loanDF():
    loan_info = new_application()
    info_dict = dict(zip(list(loans_df.columns[0:9]), loan_info))
    pprint(info_dict)

    coded_vals = [(loan_dict[i][loan_info[i]]) for i in range(len(loan_info))]
    loanInfoDf = pd.DataFrame([coded_vals], columns=loans_df.columns[:9] ,index=['NEW LOAN APPLICATION'] )
    return loanInfoDf

# pickle the random forest model to predict if a loan will repay or not
def pickle_file():
    with open('Loan_RF_Classifier.pckl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(mdl, application):
    pred = mdl.predict_proba(application)
    print('The probability that the loan will be paid is : ', (pred[:,1]))

def main():
    loan_application = loanDF()
    pred_model = pickle_file()
    predict(pred_model, loan_application)

main()


