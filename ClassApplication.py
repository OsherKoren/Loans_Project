from loanDictionary import loans_df

# Define Loan Application information/ attributes
class Application:
    def __init__(self, district=None, gender=None, ownership=None, age_range=None, country=None,
                       business_status=None, field=None, consultant=None, loan_amount_range=None):
        self.district = district
        self.gender = gender
        self.ownership = ownership
        self.age_range = age_range
        self.country = country
        self.business_status = business_status
        self.field = field
        self.consultant = consultant
        self.loan_amount_range = loan_amount_range

# Set Loan Details:
# Returns user input for business district
def set_district():
        district = input("Please Enter the business district:  Center / South / North  ")
        district = district.capitalize().replace(' ','')
        while district not in loans_df['District'].unique():
            print("NOT A VALID INPUT . PLEASE TRY AGAIN")
            district = input("Please Enter the business district:  Center / South / North  ")
            district = district.capitalize().replace(' ', '')
        else:
            print("The District was successfully typed \n")
        return district

# Returns user input of applicant gender
def set_gender():
    gender = input("Please Enter the applicant gender:  M / F ")
    gender = gender.capitalize().replace(' ','')
    while gender not in loans_df['Gender'].unique():
        print('NOT A VALID INPUT . PLEASE TRY AGAIN ')
        gender = input("Please Enter the applicant gender:  M / F ")
        gender = gender.capitalize().replace(' ', '')
    else:
        print("The Gender was successfully typed \n")
    return gender

# Returns user input of business ownership structure
def set_ownership():
    ownership = input("Please Enter the business Ownership:  One Owner / Multiple Owners ")
    ownership = ownership.title()
    while ownership not in loans_df['Ownership'].unique():
        print("NOT A VALID INPUT . PLEASE TRY AGAIN")
        ownership = input("Please Enter the business Ownership:  One Owner / Multiple Owners ")
        ownership = ownership.title()
    else:
        print("The Ownership was successfully typed \n")
    return ownership

# Returns user input for applicant age range
def set_age_range():
    age_range = input("Please Enter the Applicant age range:  20 - 30 / 30 - 40 / 40 - 50 / 50 - 60 / 60 - 70  ")
    while age_range not in loans_df['Age Range'].unique():
        print("NOT A VALID INPUT . PLEASE TRY AGAIN")
        age_range = input("Please Enter the Applicant age range:  20 - 30 / 30 - 40 / 40 - 50 / 50 - 60 / 60 - 70  ")
    else:
        print("The Age Range was successfully typed \n")
    return age_range

# Returns user input of the country of the business
def set_country():
    print('Countries . Options : \n', loans_df['Country'].unique())
    country = input("Please Enter the Country in which the business is located from the list above: ")
    country = country.title().replace('Usa','USA').replace('Uk','UK').replace('Former Ussr','Former USSR')
    while country not in loans_df['Country'].unique():
        print("NOT A VALID INPUT . PLEASE CHOOSE FROM THE LIST")
        print('Countries . Options : \n', loans_df['Country'].unique())
        country = input("Please Enter the Country in which the business is located from the list above: ")
        country = country.title().replace('Usa', 'USA').replace('Uk', 'UK').replace('Former Ussr', 'Former USSR')
    else:
        print("The country was successfully typed \n")
    return country

# Returns user input of business status (New/Old)
def set_business_status():
    business_status = input("Please Enter the business Status:  New Business / Old Business ")
    business_status = business_status.title()
    while business_status not in loans_df['Business Status'].unique():
        print("NOT A VALID INPUT . PLEASE TRY AGAIN")
        business_status = input("Please Enter the business Status:  New Business / Old Business ")
        business_status = business_status.title()
    else:
        print("The Business status was successfully typed \n")
    return business_status

# Returns user input of the business field
def set_field():
    print('Field of business. Options : \n', loans_df['Field'].unique())
    field = input("Please Enter the field of the business : ")
    field = field.title()
    while field not in loans_df['Field'].unique():
        print("NOT A VALID INPUT . PLEASE CHOOSE FROM THE LIST")
        print('Field of business. Options : \n', loans_df['Field'].unique())
        field = input("Please Enter the field of the business : ")
        field = field.title()
    else:
        print("The field of business was successfully typed \n")
    return field

# Returns user input of the consulting company that accompanies him ("Real names changed ...")
def set_consultant():
    print('Counsultant . Options : \n', loans_df['Consultant'].unique())
    consultant = input("Please Enter consultant's name in initials from the list above . example: E.D ")
    consultant = consultant.upper().replace(" ","")
    while consultant not in loans_df['Consultant'].unique():
        print("NOT A VALID INPUT . PLEASE CHOOSE FROM THE LIST")
        print('Counsultant . Options : \n', loans_df['Consultant'].unique())
        consultant = input("Please Enter consultant's name in initials from the list above . example: E.D ")
        consultant = consultant.upper().replace(" ","")
    else:
        print("The consultant initials was successfully typed \n")
    return consultant

# Returns user input of loan amount range
def set_loan_amount_range():
    print('Loan Amount Range. Options : \n',
           'Very Low (0 - 25000) \n',
           'Low (25000 - 50000) \n',
            'Medium 50000 - 75000 \n',
            'High 75000 - 100000 \n',
            'Very High 100000 - 125000 \n')
    loan_amount_range = input('Please Enter the Applicant loan amount range from the options above: Very Low, Low, Medium, High, Very High ')
    loan_amount_range = loan_amount_range.title()
    while loan_amount_range not in loans_df['Loan Amount Range'].unique():
        print("NOT A VALID INPUT . PLEASE CHOOSE FROM THE LIST")
        print('Loan Amount Range. Options : \n',
              'Very Low (0 - 25000) \n',
              'Low (25000 - 50000) \n',
              'Medium 50000 - 75000 \n',
              'High 75000 - 100000 \n',
              'Very High 100000 - 125000 \n')
        loan_amount_range = input('Please Enter the Applicant loan amount range from the options above: Very Low, Low, Medium, High, Very High ')
        loan_amount_range = loan_amount_range.title()
    else:
        print("The Loan Amount Range was successfully typed \n")
    return loan_amount_range

# Define a new loan application with it's information
def new_application():
    print('PLEASE ENTER THE LOAN APPLICATION DETAILS :')
    app = Application
    app.district = set_district()  # Define the application district
    app.gender = set_gender()  # Define the applicant gender
    app.ownership= set_ownership()  # Define the business ownership structure
    app.age_range = set_age_range()  # Define the applicant's age
    app.country = set_country() # Define the business location - country
    app.business_status = set_business_status() # Define the business status
    app.field = set_field() # Define the business field
    app.consultant = set_consultant() # Define the business consultant
    app.loan_amount_range = set_loan_amount_range() # Define the loan amount range

    app = Application(app.district, app.gender, app.ownership, app.age_range, app.country,
                      app.business_status, app.field, app.consultant, app.loan_amount_range)
    app_info = list(app.__dict__.values())

    return app_info

