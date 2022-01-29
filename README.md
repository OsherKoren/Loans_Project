# Machine Learning Classification Algorithms
https://nbviewer.jupyter.org/github/OsherKoren/Loans_Project/blob/master/loans_project.ipynb 

# Loans analysis & prediction
**Reauirements: Attached is a requirements.txt for more convenience.**

The required packages are:
```
atomicwrites==1.3.0

attrs==19.3.0

backcall==0.1.0

certifi==2020.4.5.1

chardet==3.0.4

chart-studio==1.1.0

colorama==0.4.3

cycler==0.10.0

decorator==4.4.2

idna==2.9

imbalanced-learn==0.6.2

imblearn==0.0

importlib-metadata==1.6.0

ipython==7.15.0

ipython-genutils==0.2.0

jedi==0.17.0

joblib==0.14.1

jsonschema==3.2.0

jupyter-core==4.6.3

kiwisolver==1.2.0

matplotlib==3.2.1

more-itertools==8.2.0

nbformat==5.0.6

nose==1.3.7

numpy==1.18.2

packaging==20.3

pandas==1.0.3

parso==0.7.0

pickleshare==0.7.5

plotly==4.8.1

pluggy==0.13.1

prompt-toolkit==3.0.5

psutil==5.7.0

py==1.8.1

Pygments==2.6.1

pyparsing==2.4.6

pyrsistent==0.16.0

pytest==5.4.1

python-dateutil==2.8.1

pytz==2019.3

pywin32==227

requests==2.23.0

retrying==1.3.3

scikit-learn==0.22.2.post1

scipy==1.4.1

seaborn==0.10.1

six==1.14.0

sklearn==0.0

traitlets==4.3.3

urllib3==1.25.9

wcwidth==0.1.9

xlrd==1.2.0

zipp==3.1.0
```

# The Data
```
A sample of about 1000 business loans, with about 10 features that include categorical data, 
such as gender, the industry of the business and more, as well as numerical data, such as loan amount and age.
```
# Description of the project - EDA & Prediction model of loan repayment. 
```
Analysis of doubtful and lost debts from the business loans portfolio. 
Statistical analysis by demographic, business, and sectoral characteristics. 
Then, using machine learning models, prediction of loan repayment prospects according to different parameters.
```

# Guide:
```
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
```

# How to run the program
```
Run File: loanPredict.py

Main file, main function - runs the prediction model
```
