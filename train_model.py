#   importing libraries in my project 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# loading dataset
df = pd.read_csv("dataset.csv")

# Selecting required columns and the data will come from dataset.csv in the project 
df=df[
    ["Age","MonthlyIncome","JobSatisfaction","YearsAtCompany","OverTime","Attrition"]
        
        ]

# converting Yes/No to 1/0 for easy training
df["OverTime"]=df["OverTime"].map({"Yes": 1,"No": 0})
df["Attrition"]=df["Attrition"].map({"Yes": 1,"No": 0})

# defining the features
X=df.drop("Attrition", axis=1)
y=df["Attrition"]

# train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# here training  model
model = RandomForestClassifier()
model.fit(X_train,y_train)

# Saveing the model into model.pkl file in my prject
joblib.dump(model,"model.pkl")

print("Model trained and saved as model.pkl")