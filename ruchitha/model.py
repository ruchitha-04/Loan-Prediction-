

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pickle

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

# Function to check loan eligibility based on conditions
def check_loan_eligibility(applicant_income, coapplicant_income, loan_amount, credit_history):
    """
    Check loan eligibility based on given criteria.
    """
    if (applicant_income > 2000 and
        coapplicant_income > 1000 and
        loan_amount < 200 and
        credit_history == 0):  # Credit History == 0 for eligibility
        return 'Eligible'
    else:
        return 'Not Eligible'

# Example single data point (replace these values with actual test data)
applicant_income = 4000
coapplicant_income = 2000
loan_amount = 100
credit_history = 0

# Check loan eligibility
loan_status = check_loan_eligibility(applicant_income, coapplicant_income, loan_amount, credit_history)

# Display the result
print(f"The predicted loan status is: {loan_status}")

# Test with known data
test_features = pd.DataFrame([{
    'ApplicantIncome': 1,
    'CoapplicantIncome': 1,
    'LoanAmount': 1,
    'Credit_History': 1
}])

# Predict using the model
test_prediction = model.predict(test_features)
print("Test Prediction:", 'Eligible' if test_prediction[0] == 1 else 'Not Eligible')

# Save the model
with open('loan_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training completed successfully.")
