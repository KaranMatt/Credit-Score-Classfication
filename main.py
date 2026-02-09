from fastapi import FastAPI
from pydantic import BaseModel,Field,ConfigDict
from typing import Literal
import joblib
import numpy as np
import pandas as pd


app=FastAPI(title='Credit Score Classification API')

class CustomerData(BaseModel):
    Age: int=Field(ge=0)
    Occupation: Literal['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
       'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
       'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect']
    Annual_Income: float=Field(ge=0)
    Monthly_Inhand_Salary:float=Field(ge=0)
    Num_Bank_Accounts:int=Field(ge=0)
    Num_Credit_Card:int=Field(ge=0)
    Interest_Rate:float=Field(ge=0)
    Num_of_Loan:int=Field(ge=0)
    Delay_from_due_date:int
    Num_of_Delayed_Payment:int
    Changed_Credit_Limit:float
    Num_Credit_Inquiries:int=Field(ge=0)
    Credit_Mix:Literal['Good', 'Standard', 'Bad']
    Outstanding_Debt:float
    Credit_Utilization_Ratio:float
    Credit_History_Age:int
    Payment_of_Min_Amount:Literal['No', 'NM', 'Yes']
    Total_EMI_per_month:float
    Amount_invested_monthly:float
    Payment_Behaviour:Literal['High_spent_Small_value_payments',
       'Low_spent_Large_value_payments',
       'Low_spent_Medium_value_payments',
       'Low_spent_Small_value_payments',
       'High_spent_Medium_value_payments',
       'High_spent_Large_value_payments']
    Monthly_Balance:float

    model_config=ConfigDict(populate_by_name=True)


class CreditResponse(BaseModel):
    Credit_Score:str
    Probability:float


scaler=joblib.load('models/scaler.pkl')
imputer=joblib.load('models/imputer.pkl')
encoder=joblib.load('models/encoder.pkl')
model_xgb=joblib.load('models/xgb.pkl')

@app.get('/root')
def root():
    return {'message':'Welcome to Credit Score API'}

@app.get('/health')
def health():
    return {'status':'active'}


@app.post('/predict')
def credit_score_prediction(data:CustomerData):
    input_df=pd.DataFrame([data.model_dump(by_alias=True)])

    num_cols=['Age',
  'Annual_Income',
  'Monthly_Inhand_Salary',
  'Num_Bank_Accounts',
  'Num_Credit_Card',
  'Interest_Rate',
  'Num_of_Loan',
  'Delay_from_due_date',
  'Num_of_Delayed_Payment',
  'Changed_Credit_Limit',
  'Num_Credit_Inquiries',
  'Outstanding_Debt',
  'Credit_Utilization_Ratio',
  'Credit_History_Age',
  'Total_EMI_per_month',
  'Amount_invested_monthly',
  'Monthly_Balance']
    
    cat_cols=['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

    input_df[num_cols]=imputer.transform(input_df[num_cols])
    input_df[num_cols]=scaler.transform(input_df[num_cols])

    encoded_cols=encoder.get_feature_names_out(cat_cols).tolist()
    input_df[encoded_cols]=encoder.transform(input_df[cat_cols])

    x_input=input_df[num_cols+encoded_cols]

    pred=model_xgb.predict(x_input)[0]
    prob=model_xgb.predict_proba(x_input)[0].max()

    if pred==0:
        cred='Poor'
    elif pred==1:
        cred='Standard'
    else:
        cred='Good'        

    return CreditResponse(Credit_Score=cred,Probability=prob)            
