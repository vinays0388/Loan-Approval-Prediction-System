import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# Title of the app
st.title("Loan Approval Prediction System")

# Load the dataset and preprocess
@st.cache
def load_and_preprocess_data():
    dataset = pd.read_csv("LoanApprovalPrediction.csv")
    dataset.drop(columns=['Loan_ID'], inplace=True)

    # Create a dictionary to hold label encoders for each column
    label_encoders = {}
    
    obj = (dataset.dtypes == 'object')
    for col in list(obj[obj].index):
        lb = LabelEncoder()
        dataset[col] = lb.fit_transform(dataset[col])
        label_encoders[col] = lb  # Store the fitted LabelEncoder for each column

    # Handling missing values
    dataset['Dependents'].fillna(dataset['Dependents'].mean(), inplace=True)
    dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(), inplace=True)
    dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean(), inplace=True)
    dataset['Credit_History'].fillna(dataset['Credit_History'].mean(), inplace=True)

    # Splitting data into features and target
    x = dataset.iloc[:, :-1]
    y = dataset['Loan_Status']

    return x, y, label_encoders

# Load the data and preprocess it
x, y, label_encoders = load_and_preprocess_data()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Train models
lr = LogisticRegression()
svm = SVC()
rf = RandomForestClassifier(criterion='entropy')

for clf in (lr, svm, rf):
    clf.fit(x_train, y_train)

# Taking user input via Streamlit interface
st.write("### Enter Loan Applicant's Details:")

Gender = st.selectbox('Gender', ['Male', 'Female'])
Married = st.selectbox('Married', ['Yes', 'No'])
Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox('Self Employed', ['Yes', 'No'])
ApplicantIncome = st.number_input('Applicant Income', min_value=0)
CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
LoanAmount = st.number_input('Loan Amount', min_value=0)
Loan_Amount_Term = st.number_input('Loan Amount Term', min_value=0)
Credit_History = st.selectbox('Credit History', [1.0, 0.0])
Property_Area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Reuse the original label encoders for user input
def transform_input_data(lb, value):
    """ Helper function to transform user input using pre-fitted LabelEncoder """
    return lb.transform([value])[0]

# Convert user inputs into the appropriate format for the model
user_data = {
    'Gender': transform_input_data(label_encoders['Gender'], Gender),
    'Married': transform_input_data(label_encoders['Married'], Married),
    'Dependents': int(Dependents.replace('3+', '3')),
    'Education': transform_input_data(label_encoders['Education'], Education),
    'Self_Employed': transform_input_data(label_encoders['Self_Employed'], Self_Employed),
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Property_Area': transform_input_data(label_encoders['Property_Area'], Property_Area)
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame([user_data])

# Prediction button
if st.button('Predict Loan Approval'):
    # Use RandomForestClassifier for prediction (you can choose other models too)
    prediction = rf.predict(input_df)
    loan_status = 'Approved' if prediction[0] == 1 else 'Not Approved'
    st.write(f"### Loan Status Prediction: **{loan_status}**")
