Customer Churn Prediction System
1. Introduction
Customer churn refers to the loss of customers when they stop using a company’s product or service. In highly competitive industries such as telecommunications, retaining customers is often more cost-effective than acquiring new ones. Therefore, the ability to predict churn is valuable for improving customer retention strategies and maximizing revenue. This project aims to develop a machine learning model that predicts whether a customer is likely to churn based on historical data. The system integrates data analysis, predictive modelling, and a web-based interface to allow users to input customer information and receive real-time predictions.
2. Business Understanding & Dataset
The objective of this project is to identify customers who are at risk of leaving a service. By predicting churn in advance, businesses can take proactive measures such as targeted marketing or customer engagement strategies.
The dataset used in this project is sourced from Kaggle and contains customer-related information such as:
•	Demographic details
•	Account information
•	Service usage patterns
•	Billing and payment data
The target variable is Churn, which indicates whether a customer has left the service Yes or No.
3. Methodology
3.1 Data Preprocessing
The dataset underwent several preprocessing steps to ensure data quality and suitability for machine learning:
•	Handling missing values
•	Removing duplicate records
•	Converting categorical variables into numerical format using encoding techniques
•	Feature scaling where necessary
•	Splitting the dataset into training and testing sets

3.2 Model Development
Machine learning models were developed to classify whether a customer will churn. The primary model used is:
•	Logistic Regression
An additional model may be used for comparison:
•	Decision Tree Classifier
These models were trained on the processed dataset to learn patterns associated with customer churn.
3.3 Model Evaluation
The performance of the models was evaluated using:
•	Accuracy score
•	Confusion matrix
These metrics help determine how well the model predicts churn and identifies potential errors in classification.
3.4 Model Optimization
To improve performance, the models were further refined using:
•	Hyperparameter tuning
•	Cross-validation techniques

4. Web Application
A web app was developed using the Dash framework to provide users with an interactive interface.
The application allows users to:
•	Input customer details
•	Submit data for prediction
•	Receive a result indicating whether the customer is likely to churn
The trained machine learning model is integrated into the application to generate predictions in real time.

5. Deployment
The application is deployed using Render, a cloud platform for hosting web services.
Deployment process:
to be added
