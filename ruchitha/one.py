import streamlit as st
import numpy as np
import pickle  # to load the saved model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load the pre-trained loan prediction model (you can change this path as needed)
with open('loan_model.pkl',"rb") as file:
    model = pickle.load(file)
# Load dataset
try:
    data = pd.read_csv('LOANCSVfile.csv')
except FileNotFoundError:
    print("Error: 'loan_data.csv' file not found.")
    exit()

# Display the first few rows of the dataset for debugging
print("Initial data:")
print(data.head())

# Display missing values for debugging
print("\nMissing values before imputation:")
print(data.isna().sum())

# Handle missing values
# Impute numerical values with mean
numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
imputer_num = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])

# Impute categorical values with most frequent value
categorical_cols = ['Gender', 'Dependents', 'Self_Employed', 'Loan_Status']
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Display missing values after imputation for debugging
print("\nMissing values after imputation:")
print(data.isna().sum())

# Drop rows where target variable is missing (if any)
data = data.dropna(subset=['Loan_Status'])

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Save label encoders
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

# Prepare data for training
X = data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History']]
y = data['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)




# Title for the app
st.title("Loan Prediction System")

# Introduction message
st.write("Enter the details below to check whether the loan will be approved.")

# Input fields for the user to enter their details
st.header("Applicant Details")

# Applicant Income
applicant_income = st.number_input("Applicant Income (in USD)", min_value=0)

# Co-Applicant Income
coapplicant_income = st.number_input("Co-Applicant Income (in USD)", min_value=0)

# Loan Amount
loan_amount = st.number_input("Loan Amount (in USD)", min_value=0)

# Credit History
credit_history = st.selectbox("Credit History (0: No, 1: Yes)", options=[0, 1])

# Optional: Add additional features based on your file if needed, for example:
# Dependents, Education, Property Area, etc. (if these were included in your original file)

# Add a predict button

def check_loan_eligibility(applicant_income, coapplicant_income, loan_amount, credit_history):
    """
    Check loan eligibility based on simple custom criteria.
    """
    if (applicant_income > 2000 and
        coapplicant_income > 1000 and
        loan_amount < 200 and
        credit_history == 0):  # Credit History == 0 for eligibility
        return 'Eligible'
    else:
        return 'Not Eligible'




if st.button("Predict Loan Status"):
    # Prepare the input for the model (make sure the order matches your model's training)
    # features = np.array([[applicant_income, coapplicant_income, loan_amount, credit_history]])
 


    loan_status = check_loan_eligibility(applicant_income, coapplicant_income, loan_amount, credit_history)

    test_features = pd.DataFrame([{
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Credit_History': credit_history

    }])

    # Predict using the loaded model
    prediction = model.predict(test_features)

    # Display custom eligibility based on manual criteria
    if loan_status == 'Eligible':
        st.success(f"Congrats bro your {loan_status}!!!!")
    else:
        st.error(f"Sorry bro your  {loan_status} ^-^")

    # print("Test Prediction:", 'Eligible' if prediction[0] == 1 else 'Not Eligible')

    # Show the result to the user
    