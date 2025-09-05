import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

st.header("Heart Disease Prediction Model")

dataset = pd.read_csv("Heart ml 2nd Proj/framingham.csv")
df = pd.DataFrame(dataset)

#df

print(df.isnull().sum())
df.drop(columns = ["education"] , inplace = True , axis = 1)

#print(df.isnull().sum())
df["cigsPerDay"].fillna(df["cigsPerDay"].mean() , inplace = True)
df["BPMeds"].fillna(df["BPMeds"].mean() , inplace = True)
df["totChol"].fillna(df["totChol"].mean() , inplace = True)
df["BMI"].fillna(df["BMI"].mean() , inplace = True)
df["heartRate"].fillna(df["heartRate"].mean() , inplace = True)
df["glucose"].fillna(df["glucose"].mean() , inplace = True)
print(df.isnull().sum())

x = df.drop(columns = ["TenYearCHD"])
y = df["TenYearCHD"]

#df

x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 42 , test_size = 0.2)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train , y_train)
#rfc.score(x_test , y_test)*100

li = [("lr1",LogisticRegression()) , ("svm1",SVC()) , ("dtc1",DecisionTreeClassifier()) , ("nb1",GaussianNB()) , ("nhh1",KNeighborsClassifier())]

vt = VotingClassifier(estimators=li)
vt.fit(x_train , y_train)
st.write("The model works with ",vt.score(x_test , y_test)*100," Accuracy.")

vt.predict(x_test)

Gen = st.radio("Genter:",["Male" , "Female"])
#Gender = st.number_input("Enter Gender  1 for Male and 0 for Female: ")
if Gen=="Male":
    Gender = 1
else:
    Gender = 0

Age = st.number_input("Enter the Age: ")
if Age <0 or Age > 120 :
    st.warning("Invalid Input !!")

Smoker = 0 
cigsPerDay = 0
Smok = st.radio("Are you a Smoker:",["YES","NO"])
if(Smok == "YES"):
    Smoker = 1
    cigsPerDay = st.number_input("Enter how many cigarettes you consume daily: ")
    if cigsPerDay < 0:
        st.error("Invalid Input")

BP = st.number_input("Enter BP MEDS: ")
PreSt = st.radio("Are you with prevalentStroke?",["YES","No"])
if PreSt=="Yes":
    PreStroke = 1
else:
    PreStroke = 0

preH = st.radio("Is your Blood Presure High: ",["YES","NO"])
if preH=="Yes":
    preHyp = 1
else:
    preHyp = 0

diabe = st.radio("Are you have Diabetes: ",["YES","NO"])
if diabe=="Yes":
    diabetes = 1
else:
    diabetes = 0

totChol = st.number_input("Enter Total Cholistrol: ")

sysBP = st.number_input("Enter Systolic Blood Pressure: ")

diaBP = st.number_input("Enter  Diastolic Blood Pressure: ")

BMI = st.number_input("Enter Body Mass Index: ")

HR = st.number_input("Enter Heart Rate: ")

Glu = st.number_input("Enter Glucose: ")

if st.button("Give Prediction"):
    st.balloons()
    prediction = vt.predict([[Gender , Age , Smoker , cigsPerDay , BP , PreStroke ,preHyp , diabetes , totChol , sysBP , diaBP , BMI , HR , Glu]])
    if prediction ==0:
        st.header("Prediction is : You Are Safe With No Heart Disease. ")
    else:
        st.header("Prediction is : You Have A Heart Disease. Please consult a doctor for further evaluation.")



