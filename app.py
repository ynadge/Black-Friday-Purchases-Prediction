import streamlit as st
import pickle
import numpy as np

with open("rf_model.pkl", "rb") as f:
	model = pickle.load(f)

st.title("Black Friday Purchases Predictor")

Gender = int(st.selectbox("Select Gender (0 = F, 1 = M)", ("0", "1")))

Age = int(st.selectbox("Select Age ['0-17 = 0' '18-25 = 1' '26-35 = 2' '36-45 = 3' '46-50 = 4' '51-55 = 5' '55+ = 6']",
																							('0', '1', '2', '3', '4', '5', '6')))

Occupation = int(st.selectbox("Occupation", 
																														('0', '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')))

City_Category = int(st.selectbox("Select City Category (0 = A, 1 = B, 2 = C)", ('0', '1', '2')))

Stay_In_Current_City_Years = int(st.selectbox("Select number of years in current city", ('0', '1', '2', '3', '4')))

Marital_Status = int(st.selectbox("Marital Status (0 = No, 1 = Yes)", ('0', '1')))

Product_Category_1 = int(st.selectbox("Product Category 1", 
																																						('1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')))

Product_Category_2 = int(st.selectbox("Product Category 2", ('0', '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12', '13', '14', '15', '16', '17')))

Product_Category_3 = int(st.selectbox("Product Category 3", ('0', '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12', '13', '14', '15')))

if st.button("Predict"):
	features = np.array([[Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years, Marital_Status,
                       Product_Category_1, Product_Category_2,
                       Product_Category_3]])
	predicted_purchases = model.predict(features)[0]
	st.write(f" **Predicted Purchases:** {int(predicted_purchases)}")