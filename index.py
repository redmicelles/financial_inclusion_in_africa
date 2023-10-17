# import the streamlit library 
import streamlit as st
import cat_features_interpreter
import pandas as pd
import pickle

st.sidebar.write("Predictions")

st.header("Bank Account Predictor", divider="rainbow")

country: str = st.selectbox(
    "SELECT COUNTRY",
    tuple([""] + [val for val in cat_features_interpreter.country_dict]))

year = st.number_input("YEAR", min_value=1900, value=2022)

location_type: str = st.selectbox(
    "LOCATION TYPE",
    tuple([""] + [val for val in cat_features_interpreter.location_type_dict]))

cellphone_access: str = st.radio("CELLPHONE ACCESS: ", ("Yes", "No"))

household_size = st.number_input("HOUSEHOLD SIZE", min_value=1)

age_of_respondent = st.number_input("AGE OF RESPONDENT", min_value=5, value=18)

gender: str = st.radio("GENDER OF RESPONDENT: ", ("Male", "Female"))

relationship_with_head: str = st.selectbox(
    "RELATIONSHIP WITH HEAD",
    tuple([""] + [val for val in cat_features_interpreter.relationship_with_head_dict]))

marital_status: str = st.selectbox(
    "MARITAL STATUS",
    tuple([""] + [val for val in cat_features_interpreter.marital_status_dict]))

educational_level: str = st.selectbox(
    "EDUCATIONAL LEVEL",
    tuple([""] + [val for val in cat_features_interpreter.education_level_dict]))

job_type: str = st.selectbox(
    "JOB TYPE",
    tuple([""] + [val for val in cat_features_interpreter.job_type_dict]))

def gather_inputs():
    data =  {
        "country": [cat_features_interpreter.country_dict.get(country), ],
        "year": [int(year), ],
        "location_type": [cat_features_interpreter.location_type_dict.get(location_type), ],
        "cellphone_access": [cat_features_interpreter.yes_no_dict.get(cellphone_access), ],
        "household_size": [int(household_size), ],
        "age_of_respondent": [int(age_of_respondent), ],
        "gender_of_respondent": [cat_features_interpreter.gender_dict.get(gender), ],
        "relationship_with_head": [cat_features_interpreter.relationship_with_head_dict.get(relationship_with_head), ],
        "marital_status": [cat_features_interpreter.marital_status_dict.get(marital_status), ],
        "education_level": [cat_features_interpreter.education_level_dict.get(educational_level), ],
        "job_type": [cat_features_interpreter.job_type_dict.get(job_type), ]
    }
    return pd.DataFrame(data)

def make_predictions(pred_data):
    with open("./financial_inclusion_model.pickle", "rb") as f:
        clr = pickle.load(f)
    return clr.predict(pred_data)[0]

if (st.button("Make Prediction")):
    pred_data = gather_inputs()
    if make_predictions(pred_data):
        st.success("Respondent has a bank account")
    else:
        st.error("Respondent does not have a bank account")

