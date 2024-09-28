# Loan Approval Prediction System

## Overview
This project implements a Loan Approval Prediction System using various machine learning models, including Logistic Regression, Decision Tree, and Random Forest. The application predicts whether a loan should be approved based on applicant information and other relevant features.

## Technologies Used
- **Python**: The programming language used for data processing and modeling.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For implementing machine learning algorithms.
- **Streamlit**: To create an interactive web application for user input and predictions.

## Dataset
The dataset used for this project is [LoanApprovalPrediction.csv](path/to/dataset). It contains features related to loan applications, such as:
- Gender
- Marital Status
- Number of Dependents
- Education
- Self-Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status (Target variable)

## Features
- Data cleaning and preprocessing, including handling missing values.
- Exploratory Data Analysis (EDA) using visualizations.
- Encoding categorical variables for model training.
- Model training and evaluation using Logistic Regression, Decision Tree, and Random Forest.
- An interactive Streamlit interface to input applicant details and predict loan approval.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries can be installed using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn streamlit
