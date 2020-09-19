# For data
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['lines.linewidth'] = 3
plt.rc('legend', fontsize= 20)    
plt.rc('figure', titlesize= 20) 
import seaborn as sns

# For interactive plots
import plotly.graph_objs as go
import plotly.io as pio

# For model pre-processing
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OrdinalEncoder

# For Machine learning models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# For model evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# General
import pickle
import warnings   # To ignore any warnings 
warnings.filterwarnings("ignore")

Loan_Applications = pd.read_excel('loans_file.xlsx')

Loan_Applications = Loan_Applications.loc[0:2011, :] # Row no.2012 is the sum of some columns
print(Loan_Applications.shape[0])

print(Loan_Applications.info())

# Although it seems as if there are 870 loans and thousands of 'risky' loans ('In legal treatment'/'default' and so on),
# in the excel column of 'Loan Amount' cell with no loans amount are empty, in the other columns the cell has the variable 0 as int.
# So the info counts them. Later on, I will focus only on loans and 'risky' loans with a cell value greater than 0.
print(Loan_Applications.head(4))

# From the data frame of the loans applications, I will take only the data of approved applications, that is data of loans taken
loans = Loan_Applications[Loan_Applications['Loan Amount'] > 0]
loans['Loan Amount'].describe()

# Preparing the data for analysis and later on - building machine learning model
# Create a new column-'Repaid' with the value 'N' where there is a loan amount in columns that indicate that the loan isn't repaid,
# and 'Y' otherwise.
# Create another column-'Bad Debt Amount' that takes the value of one of the columns/cases that indicate that the loan isn't repaid. 

loans['Repaid'] = 'NaN'
loans['Bad Debt Amount']=loans.loc[:,['Debt Arrangement',"In legal treatment", "Default","Delay over 90 days"]].sum(axis=1)
loans['Repaid'] = np.where(loans['Bad Debt Amount'] > 0, 'N', 'Y')

# Get a list of consultants who have submitted at least 15 approved credit applications
freq_consultants = loans.groupby('Consultant')['Consultant'].agg('count').sort_values(ascending=False)
other_consultants = freq_consultants[freq_consultants < 15]
loans['Consultant'] = loans['Consultant'].apply(lambda x: 'Other' if x in other_consultants else x)

# Create a dictionary of the stats of Former USSR in order to combine it as the same area
USSR_keys = ['Armenia', 'Azerbaijan', 'Belarus', 'Estonia', 'Georgia', 'Kazakhstan', 'Kyrgyzstan', 'Latvia',
              'Lithuania','Moldova', 'Russia', 'Tajikistan', 'Turkmenistan', 'Ukraine', 'Uzbekistan','CVCZ']
USSR_vals = ['Former USSR' for i in USSR_keys]
USSR_dict = dict(zip(USSR_keys, USSR_vals))

for k in USSR_dict:
    loans['Country'].replace(k, USSR_dict[k], inplace=True)

# Get a list of business locations (Countries) who have submitted at least 10 approved credit applications
freq_countries = loans.groupby('Country')['Country'].agg('count').sort_values(ascending=False)
other_countries = freq_countries[freq_countries < 10]
loans['Country'] = loans['Country'].apply(lambda x: 'Other' if x in other_countries else x)

# Create a new column that separate the loans into 5 categories by their amount.
# Then we can check if the "bad" loans are corrolated with the amount of a loans
bins=[0,25000,50000,75000,100000, 125000] 
groups=['Very Low','Low','Medium','High', 'Very High'] 
loans['Loan Amount Range']=pd.cut(loans['Loan Amount'],bins,labels=groups)
print(loans.head())

# Show a summary of information of 'bad' loans
bad_loans = loans[(loans["Repaid"] == 'N')]
bad_loans.describe().round(decimals=0).apply(lambda s: s.apply(lambda x: format(x, 'g')))

# Data analysis -  Exploring the data ...
# Plot total of 'risky' loans vs 'good' loans
fig = plt.figure(figsize = (6,8))
sns.countplot(x='Repaid', data=loans, palette=['#ce295e','#1c84c6'])
plt.title('Distribution Of Data By Loan Status')
plt.savefig('PaidYN.png', bbox_inches = 'tight')

# The target values are Imbalanced. I will deal with that in the section of building the machine learning model below
# Plot two distribution plot: One of the total loans, and the other only for 'bad' loans
fig = plt.figure(figsize=(10,6))
sns.distplot(loans['Loan Amount'], norm_hist=False, kde=False, color= '#0060b8')

plt.title('DISTRIBUTION OF LOAN AMOUNT', fontsize=18)
plt.gcf().set_size_inches(10, 6)
plt.tight_layout()
plt.savefig('distLoansAmount.png', bbox_inches='tight')
# Most of the loans are in the amount of 40k to 80K, where there are more than 200 loans (~23% of total of 870) are in the amount of 120K

fig = plt.figure(figsize=(10,10))
sns.distplot(loans['Bad Debt Amount'][loans['Repaid'] == 'N'], norm_hist=False, kde=False, color='red')

plt.title('DISTRIBUTION OF BAD DEBTS AMOUNT', fontsize=18)
plt.gcf().set_size_inches(10, 6)
plt.tight_layout()
plt.savefig('distDebtsAmount.png', bbox_inches='tight')
# Surprisingly, most "bad" loans are in the small amount of 20K to 75K. There are few 'bad' loans in excess of 100K.
# At this point it should be noted that while the highest amount of the loans is ~ 120,000, the amount of debt is higher because
# of high interest rates when the loan is not repaid

# Check the 'bad' debts distribution
debts_cols = ['Loan Amount', 'Debt Arrangement', 'In legal treatment', 'Default','Delay over 90 days', 'Bad Debt Amount']
count = loans[loans[debts_cols] > 0][debts_cols].count()
sum_ = loans[loans[debts_cols] > 0][debts_cols].sum().round(decimals=0)

debts = pd.DataFrame({'COUNT': count, 'SUM': sum_}, index = debts_cols).astype(int).round(0)
debts['BY SUM %'] = ((debts['SUM']/debts.loc['Loan Amount','SUM']) * 100).round(decimals=1)
debts['BY COUNT %'] = ((debts['COUNT']/debts.loc['Loan Amount','COUNT']) * 100).round(decimals=1)
print(debts)

# Spliting the data to 'good' and 'bad' loans, before plotting two donats plots 
good_value = debts['SUM'][0]- debts['SUM'][-1]
bad_value = debts['SUM'][-1]
total_values = [good_value, debts['SUM'][1], debts['SUM'][2], debts['SUM'][3], debts['SUM'][4]]

# 2 donats plots of loans data - good loans and bad loans
loans_data = [# Portfolio (inner donut)
               go.Pie(values=[good_value, bad_value],  
                      labels=['Good Loans','Bad Debts'],
                      hole=0.3,
                      sort=False,
                      textfont=dict(size=20),
                      marker={'colors':['#1c84c6','#CB4335']}),

               # Individual components (outer donut)
              go.Pie(values=total_values,
                    labels=["Good Loans",'Debt Arrangement','In legal treatment','Lost Debt','Delay over 90 days'],
                    hole=0.75,
                    sort=False,
                    textfont=dict(size=18),
                    marker={'colors':['white','#EF7F73','#F1948A','#F4A9A1','#F6BEB8']},
                    showlegend=True)]

fig =go.Figure(data=loans_data)

# Create / update the figure layout
fig.update_layout(
                  title={'text': "Loans & Bad Debts", 'y':0.9, 'x':0.9, 'font': {'size': 25}},
                  margin = dict(t=0, l=0, r=0, b=0),
                  legend=dict(font_size=25, x=1, y=0.5),
                  # Add annotations in the center of the donut pies.
                  annotations=[dict(text='Bad Debts', x=0.4, y=0.7, font_size=18, font_color = 'white', showarrow=False),
                  dict(text='Good Loans', x=0.8, y=0.5, font_size=18, font_color = 'white', showarrow=False)])

pio.write_html(fig, file='Pie_loans_fig.html', auto_open=True) # as interactive plot with .html page

# Plot the distribution of categorical columns to check for imbalance
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

loans['Gender'].value_counts(normalize=True).plot.bar(ax=axes[0][0], fontsize=20 , color='#ce295e') 
axes[0][0].set_title("GENDER", fontsize=24)

loans['Ownership'].value_counts(normalize=True).plot.bar(ax=axes[0][1], fontsize=20, color='#ce295e') 
axes[0][1].set_title("OWNERSHIP", fontsize=24)

loans['Age Range'].value_counts(normalize=True).plot.bar(ax=axes[0][2], fontsize=20,  color='#ce295e') 
axes[0][2].set_title("AGE RANGE", fontsize=24)

loans['District'].value_counts(normalize=True).plot.bar(ax=axes[1][0], fontsize=20, color='#ce295e') 
axes[1][0].set_title("DISTRICT", fontsize=24)

loans['Business Status'].value_counts(normalize=True).plot.bar(ax=axes[1][1], fontsize=20, color='#ce295e') 
axes[1][1].set_title("Business Status", fontsize=24)

loans['Loan Amount Range'].value_counts(normalize=True).plot.bar(ax=axes[1][2], fontsize=20, color='#ce295e') 
axes[1][2].set_title("Loan Amount Range", fontsize=24)

sns.despine()
plt.setp(axes[0][0].get_xticklabels(), rotation=0)
plt.setp(axes[0][1].get_xticklabels(), rotation=0)
plt.setp(axes[0][2].get_xticklabels(), rotation=25)
plt.setp(axes[1][0].get_xticklabels(), rotation=0)
plt.setp(axes[1][1].get_xticklabels(), rotation=0)
plt.setp(axes[1][2].get_xticklabels(), rotation=25)
plt.gcf().set_size_inches(20, 12)
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=.3)

plt.savefig('LoansByCategories.png', bbox_inches='tight')

# As we can see, some of the categories, such as gender, ownership, age are not balanced,
# where district, business status and loan amount range are more balanced. At this point it should be stated,
# that the loan is given to one person, for a business that can be held by him and other owners.
# Now lets explore the data in each category

# Count 'bad' loans by the loan amount range
CountLoansByRange = loans[['Loan Amount Range', 'Loan Amount']].groupby(['Loan Amount Range']).count()
CountDebtsByRange = bad_loans[['Loan Amount Range','Bad Debt Amount']].groupby(['Loan Amount Range']).count()
CountByRange = pd.concat([CountLoansByRange, CountDebtsByRange], axis=1)

CountByRange['Bad Debts %'] = (CountByRange['Bad Debt Amount']/CountByRange['Loan Amount']).astype(float).map("{:.1%}".format)
CountByRange.rename(columns={'Loan Amount': 'Loan-Count','Bad Debt Amount':'Bad Debts-Count'}, inplace=True)
print(CountByRange.head())

PaidByLoanRangeC=pd.crosstab(loans['Loan Amount Range'],loans['Repaid']) 
print(PaidByLoanRangeC.head())

fig = plt.figure(figsize=(14,6))

PaidByLoanRangeC.div(PaidByLoanRangeC.sum(1).astype(float), axis=0).plot(kind="barh",stacked=True,color=['#ce295e','#1c84c6']) 

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xlabel('Percentage', fontsize=20)
plt.ylabel('Loan Amount Range') 
plt.title('Loan Repaid: Y / N - Distribution By Loan Amount Range')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.gcf().set_size_inches(10, 6)

plt.savefig('PaidByLoanRangeFig.png', bbox_inches = 'tight')

# The highest rate of 'bad' debts is in the range of low amount of loan - 25K to 50K
# Calculate average loan amount by business status
avgLoanByBusiness = loans[['Business Status','Loan Amount']].groupby(['Business Status']).mean().astype(int)
print(avgLoanByBusiness.head())

# There seems to be no different of the mean amount of loan between new business and old business

# Count 'bad' loans by the business status
CountLoansByBusiness = loans[['Business Status', 'Loan Amount']].groupby(['Business Status']).count()
CountDebtsByBusiness = bad_loans[['Business Status','Bad Debt Amount']].groupby(['Business Status']).count()
CountByBusiness = pd.concat([CountLoansByBusiness, CountDebtsByBusiness], axis=1)

CountByBusiness['Bad Debts %'] = (CountByBusiness['Bad Debt Amount']/CountByBusiness['Loan Amount']).astype(float).map("{:.1%}".format)
CountByBusiness.rename(columns={'Loan Amount': 'Loan-Count','Bad Debt Amount':'Bad Debts-Count'}, inplace=True)
print(CountByBusiness.head())

# The percentage of bad debts by count is also 3% higher for new business than for old business. Now let's plot it
PaidByLoanBusinessC=pd.crosstab(loans['Business Status'],loans['Repaid']) 
print(PaidByLoanBusinessC.head())

fig = plt.figure(figsize=(14,6))

PaidByLoanBusinessC.div(PaidByLoanBusinessC.sum(1).astype(float), axis=0).plot(kind="barh",stacked=True, 
                                                                               color=['#ce295e','#1c84c6']) 

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xlabel('Percentage', fontsize=20)
plt.ylabel('Business Status') 
plt.title('Loan Repaid: Y / N - Distribution By Business Status')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.gcf().set_size_inches(10, 6)
plt.savefig('PaidByLoanBusinessFig.png', bbox_inches = 'tight')

# Loans by district.   First calculate the average loan amount by district
avgLoanByDistrict = loans[['District','Loan Amount']].groupby(['District']).mean().astype(int)
print(avgLoanByDistrict.head())

# The highest average loan amount is in the center - 81k and the lowest is in the south - 66k . 
# Lets plot loan amount by district
fig = plt.figure(figsize=(10,6))
sns.catplot(x="District", y='Loan Amount', kind="violin", data=loans, palette=['#ce295e','#1c84c6','#002060'])
plt.title('LOAN BY DISTRICT', fontsize=18)
plt.gcf().set_size_inches(10, 6)
plt.tight_layout()
plt.savefig('LoanByDistrict.png', bbox_inches='tight')

# 'Bad' debts by district. First by total loans amount
DebtsByDistrict = loans[['District','Loan Amount','Bad Debt Amount']].groupby(['District']).sum().astype(int)
DebtsByDistrict['Bad Debts %'] = (DebtsByDistrict['Bad Debt Amount']/DebtsByDistrict['Loan Amount']).astype(float).map("{:.1%}".format)
print(DebtsByDistrict.head())

# There is not much difference between the districts.
# Now let's look at the loans count, a more representative figure
CountLoansByDistrict = loans[['District', 'Loan Amount']].groupby(['District']).count()
CountDebtsByDistrict = bad_loans[['District','Bad Debt Amount']].groupby(['District']).count()
CountByDistrict = pd.concat([CountLoansByDistrict, CountDebtsByDistrict], axis=1)

CountByDistrict['Bad Debts %'] = (CountByDistrict['Bad Debt Amount']/CountByDistrict['Loan Amount']).astype(float).map("{:.1%}".format)
CountByDistrict.rename(columns={'Loan Amount': 'Loan-Count','Bad Debt Amount':'Bad Debts-Count'}, inplace=True)
print(CountByDistrict.head())

# The highest percentage of 'bad' debts is in the south, and the lowest is in the north.
# Lets plot it
PaidByDistrict=pd.crosstab(loans['District'],loans['Repaid']) 

fig = plt.figure(figsize=(14,6))

PaidByDistrict.div(PaidByDistrict.sum(1).astype(float), axis=0).plot(kind="barh", stacked=True, color=['#ce295e','#1c84c6'])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xlabel('Percentage', fontsize=20)
plt.ylabel('District', fontsize=20)
plt.title('Loan Repaid: Y / N - Distribution By District')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.gcf().set_size_inches(10, 6)

# plt.savefig('PaidByDistrictFig.png', bbox_inches = 'tight')
# Loans by gender.   First calculate the average loan amount by gender
avgLoanByGender = loans[['Gender','Loan Amount']].groupby(['Gender']).mean().astype(int)
print(avgLoanByGender.head())

# Man average loan amount is higher then the woman by 25% . 
# Lets plot loan amount by gender.
fig = plt.figure(figsize=(10,6))
sns.catplot(x="Gender", y='Loan Amount', kind="violin", data=loans, palette=['#ce295e','#1c84c6'])
plt.title('LOAN BY GENDER', fontsize=18)
plt.gcf().set_size_inches(10, 6)
plt.tight_layout()
plt.savefig('LoanByGender.png', bbox_inches='tight')

# Bad debts by gender. First by total loans amount
DebtsByGender = loans[['Gender','Loan Amount','Bad Debt Amount']].groupby(['Gender']).sum().astype(int)
DebtsByGender['Bad Debts %'] = (DebtsByGender['Bad Debt Amount']/DebtsByGender['Loan Amount']).astype(float).map("{:.1%}".format)
print(DebtsByGender.head())

# The percentage of bad debts is 6% higher for men than for women.
# Now let's look at the loans count, a more representative figure
CountLoansByGender = loans[['Gender', 'Loan Amount']].groupby(['Gender']).count()
CountDebtsByGender = bad_loans[['Gender','Bad Debt Amount']].groupby(['Gender']).count()
CountByGender = pd.concat([CountLoansByGender, CountDebtsByGender], axis=1)

CountByGender['Bad Debts %'] = (CountByGender['Bad Debt Amount']/CountByGender['Loan Amount']).astype(float).map("{:.1%}".format)
CountByGender.rename(columns={'Loan Amount': 'Loan-Count','Bad Debt Amount':'Bad Debts-Count'}, inplace=True)
print(CountByGender.head())

# The percentage of bad debts by count is also 6% higher for men than for women. Now let's plot it
PaidByGender=pd.crosstab(loans['Gender'],loans['Repaid']) 
print(PaidByGender.head())

fig = plt.figure(figsize=(14,6))

PaidByGender.div(PaidByGender.sum(1).astype(float), axis=0).plot(kind="barh", stacked=True, color=['#ce295e','#1c84c6'])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xlabel('Percentage', fontsize=20)
plt.ylabel('Gender', fontsize=20)
plt.title('Loan Repaid: Y / N - Distribution By Gender')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.gcf().set_size_inches(10, 6)

plt.savefig('PaidByGenderFig.png', bbox_inches = 'tight')

# Loans by status.   First calculate the average loan amount by status
avgLoanByOwnership = loans[['Ownership','Loan Amount']].groupby(['Ownership']).mean().astype(int)
print(avgLoanByOwnership.head())

# Returning resident average loan amount is higher then the immigrant by 24% . 
# Lets plot loan amount by status.
fig = plt.figure(figsize=(10,6))
sns.catplot(x="Ownership", y='Loan Amount', kind="violin", data=loans,  palette=['#ce295e','#1c84c6'])
plt.title('LOAN BY OWNERSHIP', fontsize=18)
plt.gcf().set_size_inches(10, 6)
plt.tight_layout()
plt.savefig('LoanByOwnership.png', bbox_inches='tight')

# Bad debts by status. First by total loans amount
DebtsByOwnership = loans[['Ownership','Loan Amount','Bad Debt Amount']].groupby(['Ownership']).sum().astype(int)
DebtsByOwnership['Bad Debts %'] = (DebtsByOwnership['Bad Debt Amount']/DebtsByOwnership['Loan Amount']).astype(float).map("{:.1%}".format)
print(DebtsByOwnership.head())

# The percentage of bad debts is 4% higher for One owner than for Mulitple owners.
# Now let's look at the loans count, a more representative figure
CountLoansByOwnership = loans[['Ownership', 'Loan Amount']].groupby(['Ownership']).count()
CountDebtsByOwnership = bad_loans[['Ownership', 'Bad Debt Amount']].groupby(['Ownership']).count()
CountByOwnership = pd.concat([CountLoansByOwnership, CountDebtsByOwnership], axis=1)

CountByOwnership['Bad Debts %'] = (CountByOwnership['Bad Debt Amount']/CountByOwnership['Loan Amount']).astype(float).map("{:.1%}".format)
CountByOwnership.rename(columns={'Loan Amount': 'Loan-Count','Bad Debt Amount':'Bad Debts-Count'}, inplace=True)
print(CountByOwnership.head())

# The percentage of bad debts by count is also 4% higher for One owner than for Multiple owners. Now let's plot it
PaidByOwnership=pd.crosstab(loans['Ownership'],loans['Repaid']) 
print(PaidByOwnership.head())

fig = plt.figure(figsize=(14,6))

PaidByOwnership.div(PaidByOwnership.sum(1).astype(float), axis=0).plot(kind="barh", stacked=True, color=['#ce295e','#1c84c6'])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xlabel('Percentage', fontsize=20)
plt.ylabel('Ownership', fontsize=20)
plt.title('Loan Repaid: Y / N - Distribution By Status')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.gcf().set_size_inches(10, 6)

plt.savefig('PaidByOwnershipFig.png', bbox_inches = 'tight')

# Loans by age range.   First calculate the average loan amount by age range
avgLoanByAge = loans[['Age Range','Loan Amount']].groupby(['Age Range']).mean().astype(int)
print(avgLoanByAge.head())

# The highest average loan amount is at the age range of 60-70  = 83k and the lowest is in the age range of 20-30 = 68k . 
# Lets plot loan amount by age range
fig = plt.figure(figsize=(10,8))
sns.jointplot(x='Age',y='Loan Amount',data=loans ,kind='kde', color='#ce295e')
plt.title('LOAN BY AGE', fontsize=18, loc='left')
plt.gcf().set_size_inches(10, 8)
plt.tight_layout()
plt.savefig('LoanByAge.png', bbox_inches='tight')

fig = plt.figure(figsize=(12, 6))
sns.boxplot(x='Age Range', y='Loan Amount', data=loans, order=["20 - 30", "30 - 40", "40 - 50","50 - 60","60 - 70"],
                               palette=['#bfbfbf','#44546a','#ce295e','#1c84c6','#002060'])

plt.title('LOAN BY AGE-RANGE', fontsize=20)
plt.gcf().set_size_inches(12, 6)
plt.tight_layout()
plt.savefig('LoanByAgeRange.png', bbox_inches='tight')

# Bad debts by age range. First by total loans amount
DebtsByAge = loans[['Age Range','Loan Amount','Bad Debt Amount']].groupby(['Age Range']).sum().astype(int)
DebtsByAge['Bad Debts %'] = (DebtsByAge['Bad Debt Amount']/DebtsByAge['Loan Amount']).astype(float).map("{:.1%}".format)
print(DebtsByAge.head())

# The highest percentage of 'bad' debts is in the age range of 40-50 , and the lowest is in the age range of 60-70
# Lets plot it
fig = plt.figure(figsize=(10,8))
sns.jointplot(x='Age' ,y='Bad Debt Amount',data=bad_loans, color='#ce295e', kind='hex')
plt.title('DISTRIBUTION OF BAD \n LOANS BY AGE', loc='left')
plt.gcf().set_size_inches(10, 8)
plt.tight_layout()
plt.savefig('BadDeptsByAge.png', bbox_inches = 'tight')

# Now let's look at the loans count, a more representative figure
CountLoansByAge = loans[['Age Range', 'Loan Amount']].groupby(['Age Range']).count()
CountDebtsByAge = bad_loans[['Age Range', 'Bad Debt Amount']].groupby(['Age Range']).count()
CountByAge = pd.concat([CountLoansByAge, CountDebtsByAge], axis=1)

CountByAge['Bad Debts %'] = (CountByAge['Bad Debt Amount']/CountByAge['Loan Amount']).astype(float).map("{:.1%}".format)
CountByAge.rename(columns={'Loan Amount': 'Loan-Count','Bad Debt Amount':'Bad Debts-Count'}, inplace=True)
print(CountByAge.head())

# The highest percentage of 'bad' debts is in the age range of 20-30 , and the lowest is in the age range of 60-70
# Lets plot it
PaidByAge=pd.crosstab(loans['Age Range'],loans['Repaid']) 
print(PaidByAge.head())

fig = plt.figure(figsize=(14,6))

PaidByAge.div(PaidByAge.sum(1).astype(float), axis=0).plot(kind="barh", stacked=True, color=['#ce295e','#1c84c6'])

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xlabel('Percentage', fontsize=20)
plt.ylabel('Age', fontsize=20)
plt.title('Loan Repaid: Y / N - Distribution By Age')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.gcf().set_size_inches(10, 6)

plt.savefig('PaidByAgeFig.png', bbox_inches = 'tight')

# Loans by business location - by country.
# Lets plot the major countries that received loans
fig = plt.figure(figsize=(12,8))
loans['Country'].value_counts().head(12).plot(kind='barh', color='#ce295e')
plt.xlabel('Loans - Count', fontsize=20)
plt.title('LOANS BY COUNTRIES - COUNT')
plt.gcf().set_size_inches(12, 8)
plt.tight_layout()
plt.savefig( 'LoansByCountry.png', bbox_inches = 'tight')

# The top 3 areas are: Former USSR, USA and France. 
# Lets analyze 'bad' debts by country. By loans count.
CountLoansByCountry = loans[['Country', 'Loan Amount']].groupby(['Country']).count()
CountDebtsByCountry = bad_loans[['Country', 'Bad Debt Amount']].groupby(['Country']).count()
CountByCountry = pd.concat([CountLoansByCountry, CountDebtsByCountry], axis=1).sort_values(by='Loan Amount', ascending=False)
CountByCountry.rename(columns={'Loan Amount': 'Loans - Count', 'Bad Debt Amount': 'Bad Debts - Count'}, inplace=True)

CountByCountry['Bad Debts %'] = (CountByCountry['Bad Debts - Count']/CountByCountry['Loans - Count'] * 100).astype(float).round(1)

# Prepare the data for interactive world map plot
worldf = CountByCountry.rename(index={'Former USSR': 'Russia'}).drop('Other').sort_values('Bad Debts %', ascending = False).reset_index()
print(worldf.head())

# The highest percentage of 'bad' debts is in Argentina , and the lowest is in Italy.
# Plot an interactive world map with the data
world_data = dict(
                   type = 'choropleth',
                   colorscale =  'reds',
                   reversescale = False,
                   locations = worldf['Country'],
                   locationmode = "country names",
                   z = worldf['Bad Debts %'],
                   text = worldf['Country'],
                   colorbar = {'title' : '% Bad Debts'}) 

layout = dict(title = '% Bad Debts By Country',
                geo = dict(showframe = False, projection = {'type':'natural earth'}))

choromap = go.Figure(world_data ,layout)

pio.write_html(choromap, file='map_fig.html',  auto_open=True) # as interactive plot with .html page

# Bad debts by field of business. By loans count.
CountLoansByFields = loans[['Field', 'Loan Amount']].groupby(['Field']).count()
CountDebtsByFields = bad_loans[['Field', 'Bad Debt Amount']].groupby(['Field']).count()
CountByField = pd.concat([CountLoansByFields, CountDebtsByFields], axis=1).sort_values(by='Loan Amount', ascending=False)
CountByField.rename(columns={'Loan Amount': 'Loans - Count', 'Bad Debt Amount': 'Bad Debts - Count'}, inplace=True)

CountByField['Bad Debts %'] = (CountByField['Bad Debts - Count']/CountByField['Loans - Count'] * 100).astype(float).round(1)
print(CountByField.head())

bad_fields = CountByField[['Bad Debts %']].sort_values('Bad Debts %', ascending=False)
print(bad_fields.head())

# The highest percentage of 'bad' debts is in the fiels of 'food and restaurants', and the lowest is in 'Attractions'.
# Lets plot the data.
field_data = go.Figure(data=[go.Scatter(
                             x=bad_fields.index,
                             y=bad_fields['Bad Debts %']/100,
                             mode='markers',
                             marker=dict(
                                         color=['#ce295e', '#ce295e','#ce295e','#ce295e',
                                                '#002060','#002060','#002060','#002060',
                                                '#0060b8','#0060b8','#0060b8','#0060b8',
                                                '#44546a','#44546a','#44546a','#44546a',
                                                '#7f7f7f','#7f7f7f','#7f7f7f','#7f7f7f'],
                                         size = list(range(105, 15, -5)),
                                         showscale=False
                                          ))])

fig =go.Figure(data=field_data)

fig.update_layout(
                  autosize=False, width=1000, height=800,
                  title={'text':'<b>% Of Bad Debts By Fields<b>', 'y':0.9, 'x':0.5, 'font': {'size': 20}},
                  xaxis = go.layout.XAxis(tick0 = 2,dtick = 1, tickangle=40, showgrid=False),
                  yaxis = go.layout.YAxis(tickformat = '%', showgrid=False),
                  font=dict(family="Courier New, monospace",size=18),
                  margin=dict(l=0, r=50, b=250, t=35, pad=4))

pio.write_html(fig, file='Scatter_field_fig.html', auto_open=True) # as interactive plot with .html page

# Bad debts by business consultant. By loans count.
CountLoansByConsultants = loans[['Consultant', 'Loan Amount']].groupby(['Consultant']).count()
CountDebtsByConsultants = bad_loans[['Consultant', 'Bad Debt Amount']].groupby(['Consultant']).count()
CountByConsultants= pd.concat([CountLoansByConsultants, CountDebtsByConsultants], axis=1).sort_values(by='Loan Amount', ascending=False)
CountByConsultants.rename(columns={'Loan Amount': 'Loans - Count', 'Bad Debt Amount': 'Bad Debts - Count'}, inplace=True)

CountByConsultants['Bad Debts %'] = (CountByConsultants['Bad Debts - Count']/CountByConsultants['Loans - Count'] * 100).astype(float).round(1)
CountByConsultants = CountByConsultants.iloc[:15]
print(CountByConsultants.head())

bad_consultants = CountByConsultants[['Bad Debts %']].sort_values('Bad Debts %', ascending=False)
print(bad_consultants.head())

# Lets plot the data
consultants_data = go.Figure(data=[go.Scatter(
                                   x=bad_consultants.index,
                                   y=bad_consultants['Bad Debts %']/100,
                                   mode='markers',
                                   marker=dict(
                                              color=['#ce295e', '#ce295e','#ce295e','#ce295e',
                                                     '#002060','#002060','#002060','#002060',
                                                     '#0060b8','#0060b8','#0060b8','#0060b8',
                                                     '#44546a','#44546a','#44546a','#44546a',
                                                     '#7f7f7f','#7f7f7f','#7f7f7f','#7f7f7f'],
                                              size = list(range(100, 15, -5)),
                                              showscale=False
                                               ))])

fig =go.Figure(data=consultants_data)

fig.update_layout(
                  autosize=False, width=1000, height=800,
                  title={'text':'<b> % Of Bad Debts By Consultants <b>', 'y':0.9, 'x':0.5, 'font': {'size': 25}},
                  xaxis = go.layout.XAxis(tick0 = 2,dtick = 1, tickangle=40),
                  yaxis = go.layout.YAxis(tickformat = '%'),
                  font=dict(family="Courier New, monospace",size=18),
                  margin=dict(l=50, r=20, b=200, t=35, pad=4))
        
pio.write_html(fig, file='Scatter_consultants_fig.html', auto_open=False) # as interactive plot with .html page

# After exploring and analyzing the data, I will use machine learning algorithm on the data and try to predict
# if a loan will be repaid or is it a 'risky' loan.
# First, copy the original data frame, and take only 'object' and 'category' columns type to train on
categ_data = loans.select_dtypes(include=['object','category']).copy()
print(categ_data.info())

# Check to see if there are null values. I didn't encounter null values along the data analysis, but I will check anyway
print(categ_data.isnull().sum())

# Data Preparation
# Change the object and categorical columns types into numeric, before training the model
# Keep the source of categorical data, and create a copy for different encoding methods
encoded_data = categ_data.copy()
# We have two ordinal categorical. Lets see the values. 
print('Loan Amount Range: ', groups)
print ('Age Range: ', encoded_data['Age Range'].unique())

# Keep the order of the categories with numeric featuers 
encoder = OrdinalEncoder()
encoded_data[['Loan Amount Range', 'Age Range']] = encoder.fit_transform(encoded_data[['Loan Amount Range', 'Age Range']]).astype(int)

# The rest of the columns are object type.
obj_cols = encoded_data.drop(['Age Range', 'Loan Amount Range'], axis=1).columns

encoded_data[obj_cols] = encoded_data[obj_cols].apply(lambda x: pd.factorize(x)[0])
print(encoded_data.head())

encoded_X = encoded_data.drop(['Repaid'], axis=1)
encoded_y = encoded_data['Repaid']

# Pre check
encoded_X.shape[0] == encoded_y.shape[0]

# Plot and check if there is correlation between the categories
fig = plt.figure(figsize=(8,6))
corr = encoded_data.corr()
sns.heatmap(corr, cmap="coolwarm_r")
plt.title('Correlation Plot')
plt.savefig('corr.png', bbox_inches = 'tight')

# It seems that there is no correlation between the categories

X_train, X_test, y_train, y_test = train_test_split(encoded_X, encoded_y, test_size = 0.2, random_state = 42 )

# Check the sets shape
print('Training shape: ', X_train.shape)
print('Testing shape: ', X_test.shape)

# Resample the data to get balanced target values
# First, recall the unbalanced target data
count_Repaid_N, count_Repaid_Y = encoded_data.Repaid.value_counts()
print(count_Repaid_N, count_Repaid_Y)

# Separate by target value
Repaid_N = encoded_data[encoded_data['Repaid'] == 0]
Repaid_Y = encoded_data[encoded_data['Repaid'] == 1]

# Use random over-sampling to balance the target class of 'N' to the 'Y' before training the data
ros = RandomOverSampler(random_state=1)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Count and recheck if now the target values are balanced
from collections import Counter 
print(sorted(Counter(y_train).items()))

# Train different models

# Print results
def print_report(m_name, trn_accuracy, tst_accuracy, tst_pred):
   print("-"*60)
   print(m_name)
   print ('Train Accuracy: {:.2%}'.format(trn_accuracy))
   print ('Test Accuracy: {:.2%}'.format(tst_accuracy))
   if trn_accuracy > 1.2 * tst_accuracy :
       print(' O V E R  F I T T I N G ')
   print('\n')
   print('\n', classification_report(y_test, tst_pred, target_names=["NO", "YES"]))
   print('\n')

def plot_confusion_matrix(c_m, m_name):
    fig, ax = plt.subplots()
    sns.heatmap(c_m, annot=True, cmap="coolwarm_r", linewidths=1, annot_kws={"size": 30},  fmt="d",
                                      xticklabels=['NO','YES'], yticklabels=['NO','YES'])

    plt.title('Confusion Matrix: {:s}'.format(m_name))
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(m_name+'.png', bbox_inches='tight')

# Save the model as a pickl file for predictions
def pickle_model(name, mdl):
    with open(name+'.pckl', 'wb') as f:
        pickle.dump(mdl, f)

models_dict = {'Logistic regression': LogisticRegression(random_state=42),
              'Decision Tree' : DecisionTreeClassifier(random_state=42),
               'Random Forest' : RandomForestClassifier(random_state=42),
               'SVM' : svm.LinearSVC(random_state=42) }

# Train models
def ml(mdl_dict):
    summary_dict = {}
    for k in mdl_dict:
        model_name = k
        model = mdl_dict[k]
        model.fit(X_train, y_train)
        
        # Create reports to evaluate the models
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, test_pred)
        train_accu = accuracy_score(y_train, train_pred)
        test_accu = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, average='binary')
        recall = recall_score(y_test, test_pred, average='binary')
        score = f1_score(y_test, test_pred, average='binary')
        
        summary_dict[k] = [test_accu, precision, recall, score]
        
        # Print , plot and pickle
        print_report(model_name, train_accu, test_accu, test_pred)
        plot_confusion_matrix(cm, model_name)
        pickle_model(model_name, model)
        
    # Summary report for comparing models results
    summary = pd.DataFrame(summary_dict.values(), columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                           index=summary_dict.keys()).round(2)
    print ('Summary Of Models Scores ')
    print ('-'*60)
    print (summary)


# Train different models
ml(models_dict)

# Best model seems to be Random Forest

# Try another model -  KNN
def print_knn(k, precision, recall, score, trn_accuracy, tst_accuracy, tst_pred, cm, mdl):
    print('Summary Of KNN Model Scores with K = {:d}'.format(k))      
    print ('Precision {:.2%}'.format(precision))
    print ('Recall  {:.2%}'.format(recall))
    print ('F1 Score {:.2%}'.format(score))
    print('\n')
    print ('Train Accuracy: {:.2%}'.format(trn_accuracy))
    print ('Test Accuracy: {:.2%}'.format(tst_accuracy))
    if trn_accuracy > 1.2 * tst_accuracy :
        print('O V E R  F I T T I N G ')
    print('\n', classification_report(y_test, tst_pred, target_names=["NO", "YES"]))
    
    plot_confusion_matrix(cm, 'KNN')
    pickle_model('KNN', mdl)

# Knn model with k of minimum error rate
def knn_model(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    
    # Create reports to evaluate the models
    train_pred = knn.predict(X_train)
    test_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, test_pred)
    train_accu = accuracy_score(y_train, train_pred)
    test_accu = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, average='binary')
    recall = recall_score(y_test, test_pred, average='binary')
    score = f1_score(y_test, test_pred, average='binary')
    
    print_knn(k, precision, recall, score, train_accu, test_accu, test_pred, cm, knn)

def plot_error_rate(err_rates):
    keys , vals = zip(*sorted(err_rates.items()))
    min_k = min(err_rates, key=err_rates.get)
    
    print('K of minimum error rate is: {:d}'.format(min_k))
    
    plt.figure(figsize=(10,6))
    plt.plot(keys, vals, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')   
    
    knn_model(min_k)

# Use the elbow method to pick a good K Value:
def elbow_method(a, b, c):
    error_rate = {}
    for k in range(a, b, c):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        pred_k = knn.predict(X_test)
        error_rate[k] = np.mean(pred_k != y_test)
               
    plot_error_rate(error_rate)

elbow_method(1, 30, 2)

# KNN model doesn't give better results than the random forest model
# I will stick to the Random Forest model, and try to improve it

def print_model(tst_accuracy, precision, recall, score, cm, name, mdl):
    print('Summary Of Random Forest Model Scores') 
    print ('Accuracy {:.2%}'.format(tst_accuracy))
    print ('Precision {:.2%}'.format(precision))
    print ('Recall  {:.2%}'.format(recall))
    print ('F1 Score {:.2%}'.format(score))
    print('\n')
    
    plot_confusion_matrix(cm, name)
    pickle_model(name,mdl )

# Train a single model
def ml_model(model_name, model):
    model.fit(X_train, y_train)
    # Create reports to evaluate the models
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, test_pred)
    train_accu = accuracy_score(y_train, train_pred)
    test_accu = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, average='binary')
    recall = recall_score(y_test, test_pred, average='binary')
    score = f1_score(y_test, test_pred, average='binary')
    
    print_model(test_accu, precision, recall, score, cm, model_name, model)

rf = RandomForestClassifier(random_state=42, n_estimators=100)

# Random Search for optimal hyper-parameters

# Set parameters and distributions to sample from
from scipy.stats import randint  # Initialize random values for the parameters below.
param_dist = { 
    'n_estimators': randint(100, 200),
    'max_features': randint(1, 10.5),
    'max_depth' : [2,3,4,5, None],
    'min_samples_split' : randint(2,11),
    'min_samples_leaf' : randint(1,11),
    'criterion' :['gini', 'entropy'],
    'bootstrap' : [True, False],
}

# Search randomly with parameters above. Takes some time. after getting the result, I marked it as a comment
"""
random_rf  = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_rf.fit(X, y)
from pprint import pprint # For better printing
pprint('Best parameters for the the Random Forest Tree model are  {}'.format(random_rf.best_params_))
"""

print('Fitting 5 folds for each of 100 candidates, totalling 500 fits')
print('Best parameters for the the Random Forest Tree model are')
print('n_estimators: 101')
print('max_features: 2')
print('max_depth: None')
print('min_samples_split: 7')
print('min_samples_leaf: 8')
print('criterion: entropy')
print('bootstrap: True')

# Tune Random Forest model with best parameters and compare the results with the default parameters in the model
tuned_rf = RandomForestClassifier(n_estimators=101,
                                 criterion='entropy',
                                 max_depth=None,
                                 min_samples_split=7,
                                 min_samples_leaf=8,
                                 max_features=2,
                                 bootstrap='True',
                                 random_state=42)
print ('Random Forest Model with default parameters')
print ('-'*50)
default_rf = ml_model('Loan_RF_Classifier', rf) # With default parameters
print ('Random Forest Model with tuned parameters')
print ('-'*50)
tunedRF = ml_model('Tuned_Loan_RF_Classifier', tuned_rf)


# Default parameters produce better results

# Which are the most important features in the model

feature_importances = pd.DataFrame(rf.feature_importances_, index=encoded_data.columns.drop('Repaid'),
                                   columns=['Importance']).sort_values('Importance', ascending=False)

fig = plt.figure(figsize=(10,6))
ax = sns.barplot(x=feature_importances.index, y='Importance', data=feature_importances, palette='Blues_r', saturation=0.4)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.savefig('importanceFeatures.png', bbox_inches = 'tight')

encoded_data = encoded_data.rename(columns={'District':'District_Code', 'Gender':'Gender_Code', 'Ownership':'Ownership_Code',
                            'Age Range': 'Age_Code', 'Country':'Country_Code', 'Business Status':'Business_Code',
                            'Field':'Field_Code', 'Consultant':'Consultant_Code','Loan Amount Range':'Loan_Amount_Code'})

loans_df = pd.concat([categ_data,encoded_data],axis=1).drop('Repaid', axis=1)
loans_df.head()

loans_df.to_pickle('loans_df')