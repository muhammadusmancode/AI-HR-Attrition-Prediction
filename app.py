import streamlit as st
import joblib
import numpy as np

# Loading trained model and making the web UI using Streamlit
model=joblib.load("model.pkl")

st.title("AI HR Attrition Prediction System")

st.write("Enter employee details so we can check if he is willing to leave or stay ")

age=st.number_input("Age",18,65,30)

income=st.number_input("Monthly Income",1000,20000,5000)
job_satisfaction=st.slider("Job Satisfaction (1 = Low, 4 = High)",1,4,3)
years_at_company=st.number_input("Years at Company", 0, 40, 5)

overtime=st.selectbox("OverTime", ["No","Yes"])
overtime_val=1 if overtime=="Yes" else 0

if st.button("Predict"): 
    input_data=np.array(
        [[age, income,job_satisfaction,years_at_company,overtime_val]
         ]
        )
    prediction=model.predict(input_data)[0]

    if prediction==1:
        st.error("High Risk:  Employee wants to leave ")
    else:
        st.success("Low Risk:  Employee wants to stay ")


