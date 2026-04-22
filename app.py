import joblib
import dash_mantine_components as dmc
import dash_ag_grid as dag
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State, no_update

df = pd.read_csv('clean_dataset.csv')
logreg = joblib.load('logreg.pkl')
scaler = joblib.load('scaler.pkl')
dtree = joblib.load('dtree_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

app = Dash()
server = app.server  # <-- exposed for gunicorn

df['Churn'] = df['Churn'].map({1: 'Churned', 0: 'Retained'})

app.layout = dmc.MantineProvider(
    dmc.Container([

        dmc.Title('Customer Churn Analysis Dashboard', c='teal', order=1),

        dmc.Select(
            label='Select Model',
            data=['Logistic Regression', 'Decision Tree'],
            value='Logistic Regression',
            id='model-select'
        ),

        dmc.Select(
            label='Contract Type',
            data=['Month-to-month', 'One year', 'Two year'],
            id='contract'
        ),

        dmc.NumberInput(label='Tenure (months)', value=12, id='tenure'),
        dmc.NumberInput(label='Monthly Charges', value=50, id='monthly-charges'),
        dmc.NumberInput(label='Total Charges', value=0, id='total_charges'),

        dmc.RadioGroup(
            label='Select Gender',
            id='gender',
            children=[
                dmc.Radio('Male', value='Male'),
                dmc.Radio('Female', value='Female')
            ]
        ),

        dmc.RadioGroup(
            label='Senior Citizen yes/no?',
            id='senior_citizen',
            children=[
                dmc.Radio('Yes', value='Yes'),
                dmc.Radio('No', value='No')
            ]
        ),

        dmc.RadioGroup(
            label='Partner yes/no?',
            id='partner',
            children=[
                dmc.Radio('Yes', value='Yes'),
                dmc.Radio('No', value='No')
            ]
        ),

        dmc.RadioGroup(
            label='Dependents yes/no?',
            id='dependents',
            children=[
                dmc.Radio('Yes', value='Yes'),
                dmc.Radio('No', value='No')
            ]
        ),

        dmc.RadioGroup(
            label='Phone Service yes/no?',
            id='phone_service',
            children=[
                dmc.Radio('Yes', value='Yes'),
                dmc.Radio('No', value='No')
            ]
        ),

        html.Div(
            id='multiplelinesContainer',
            children=[
                dmc.RadioGroup(
                    label='Multiple Lines yes/no?',
                    id='multiplelines',
                    children=[
                        dmc.Radio('Yes', value='Yes'),
                        dmc.Radio('No', value='No')
                    ]
                )
            ]
        ),

        dmc.Select(
            label='Internet Service',
            data=['DSL', 'Fiber optic', 'No'],
            id='internet_service'
        ),

        html.Div(
            id='NeedInternetContainer',
            children=[
                dmc.RadioGroup(
                    label='Online Security yes/no?',
                    id='onlinesecurity',
                    children=[
                        dmc.Radio('Yes', value='Yes'),
                        dmc.Radio('No', value='No')
                    ]
                ),

                dmc.RadioGroup(
                    label='Online Backup yes/no?',
                    id='onlinebackup',
                    children=[
                        dmc.Radio('Yes', value='Yes'),
                        dmc.Radio('No', value='No')
                    ]
                ),

                dmc.RadioGroup(
                    label='Device Protection yes/no?',
                    id='deviceprotection',
                    children=[
                        dmc.Radio('Yes', value='Yes'),
                        dmc.Radio('No', value='No')
                    ]
                ),

                dmc.RadioGroup(
                    label='Tech Support yes/no?',
                    id='techsupport',
                    children=[
                        dmc.Radio('Yes', value='Yes'),
                        dmc.Radio('No', value='No')
                    ]
                ),

                dmc.RadioGroup(
                    label='Streaming TV yes/no?',
                    id='streamingTV',
                    children=[
                        dmc.Radio('Yes', value='Yes'),
                        dmc.Radio('No', value='No')
                    ]
                ),

                dmc.RadioGroup(
                    label='Streaming Movies yes/no?',
                    id='streamingMovies',
                    children=[
                        dmc.Radio('Yes', value='Yes'),
                        dmc.Radio('No', value='No')
                    ]
                )
            ]
        ),

        dmc.RadioGroup(
            label='Paperless billing yes/no?',
            id='paperlessBill',
            children=[
                dmc.Radio('Yes', value='Yes'),
                dmc.Radio('No', value='No')
            ]
        ),

        dmc.Select(
            label='Payment Method',
            data=['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit card'],
            id='paymentMethod'
        ),

        dmc.Button('Predict Churn', id='predict-btn', color='teal', mt='md'),

        html.Div(
            id='prediction-output',
            style={'marginTop': '20px', 'fontSize': '20px'}
        )
    ])
)


@callback(
    Output('multiplelinesContainer', 'style'),
    Input('phone_service', 'value'),
)
def toggle_multiple_lines(phone_val):
    if phone_val == 'Yes':
        return {'display': 'block', 'marginTop': '10px'}
    return {'display': 'none'}


@callback(
    Output('NeedInternetContainer', 'style'),
    Input('internet_service', 'value')
)
def toggle_online_backup(internet_val):
    if internet_val == 'Fiber optic' or internet_val == 'DSL':
        return {'display': 'block', 'marginTop': '10px'}
    return {'display': 'none'}


@callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('model-select', 'value'),
    State('contract', 'value'),
    State('tenure', 'value'),
    State('monthly-charges', 'value'),
    State('senior_citizen', 'value'),
    State('gender', 'value'),
    State('partner', 'value'),
    State('total_charges', 'value'),
    State('dependents', 'value'),
    State('phone_service', 'value'),
    State('multiplelines', 'value'),
    State('internet_service', 'value'),
    State('onlinesecurity', 'value'),
    State('onlinebackup', 'value'),
    State('deviceprotection', 'value'),
    State('techsupport', 'value'),
    State('streamingTV', 'value'),
    State('streamingMovies', 'value'),
    State('paperlessBill', 'value'),
    State('paymentMethod', 'value')
)
def predict_churn(n_clicks, model_chosen, contract, tenure, monthly_charges, gender,
                  senior_citizen, partner, total_charges, dependents, phone_service,
                  multiplelines, internet_service, onlinesecurity, onlinebackup,
                  deviceprotection, techsupport, streamingTV, streamingMovies,
                  paperlessBill, paymentMethod):
    if n_clicks is None:
        return no_update

    model = logreg if model_chosen == 'Logistic Regression' else dtree

    input_data = pd.DataFrame([{col: 0 for col in feature_columns}])

    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges

    if contract == 'One year':
        input_data['Contract_one year'] = 1
    elif contract == 'Two year':
        input_data['Contract_two year'] = 1

    if gender == 'Male':
        input_data['gender_male'] = 1

    if senior_citizen == 'Yes':
        input_data['SeniorCitizen'] = 1

    if partner == 'Yes':
        input_data['Partner_yes'] = 1

    if dependents == 'Yes':
        input_data['Dependents_yes'] = 1

    if phone_service == 'Yes':
        input_data['PhoneService_yes'] = 1
    else:
        input_data['MultipleLines_no phone service'] = 1

    if multiplelines == 'Yes':
        input_data['MultipleLines_yes'] = 1

    if internet_service == 'Fiber optic':
        input_data['InternetService_fiber optic'] = 1
    elif internet_service == 'No':
        input_data['InternetService_no'] = 1
        input_data['OnlineSecurity_no internet service'] = 1
        input_data['OnlineBackup_no internet service'] = 1
        input_data['DeviceProtection_no internet service'] = 1
        input_data['TechSupport_no internet service'] = 1
        input_data['StreamingTV_no internet service'] = 1
        input_data['StreamingMovies_no internet service'] = 1

    if onlinebackup == 'Yes':
        input_data['OnlineBackup_yes'] = 1

    if deviceprotection == 'Yes':
        input_data['DeviceProtection_yes'] = 1

    if techsupport == 'Yes':
        input_data['TechSupport_yes'] = 1

    if streamingTV == 'Yes':
        input_data['StreamingTV_yes'] = 1

    if streamingMovies == 'Yes':
        input_data['StreamingMovies_yes'] = 1

    if paperlessBill == 'Yes':
        input_data['PaperlessBilling_yes'] = 1

    if paymentMethod == 'Credit card':
        input_data['PaymentMethod_credit card (automatic)'] = 1
    elif paymentMethod == 'Electronic Check':
        input_data['PaymentMethod_electronic check'] = 1
    elif paymentMethod == 'Mailed Check':
        input_data['PaymentMethod_mailed check'] = 1

    if model_chosen == 'Logistic Regression':
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    return '🔴 Will Churn' if prediction[0] == 1 else '🟢 Will Not Churn'


if __name__ == '__main__':
    app.run(debug=True, port=8051)
