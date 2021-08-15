# import libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier


def main():
    st.title('Stroke Prediction')

    st.markdown("""
    	This app predicts whether a patient is likely to get a stroke or not using this 
        [dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) which consists of input parameters like 
        gender, age, various diseases, and smoking status leveraging a regularizing gradient boosting model. 
	""")

    st.markdown("## Prediction")

    st.markdown("""
        To make a prediction, select features on the sidebar and click on Classify below.
        """)

    st.sidebar.markdown('## Select input features')

    # Load cleaned dataset
    data = pd.read_csv('data/stroke-data.csv')
    data = data.drop(columns=['stroke', 'id'])

    # replace 0, and 1 with no and yes respectively in hypertension and
    # heart disease column
    mapper = {0: 'No', 1: 'Yes'}

    def encode(val):
        return mapper[val]

    for col in ['hypertension', 'heart_disease']:
        data[col] = data[col].apply(encode)

    # set feature values 
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.number_input('Age', 1, 100, 24)
    hypertension = st.sidebar.selectbox('Hypertensive', ('Yes', 'No'))
    heart_disease = st.sidebar.selectbox('Heart disease', ('Yes', 'No'))
    married = st.sidebar.selectbox('Ever married', ('Yes', 'No'))
    work = st.sidebar.selectbox('Work type', ('Private', 'Self-employed',
                                              'Govt_job', 'children', 'Never_worked'))
    residence = st.sidebar.selectbox('Type of place of residence', ('Urban', 'Rural'))
    glucose_level = st.sidebar.number_input(label='Average glucose level', min_value=53.0,
                                            max_value=275.0, value=66.1, step=0.1)
    bmi = st.sidebar.number_input('BMI', 9.0, 99.0, 24.9, 0.1)
    smoke = st.sidebar.selectbox('Smoking status', ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))

    # make a list of features 
    features = [gender, age, hypertension, heart_disease, married, work, residence,
                glucose_level, bmi, smoke]

    # create dataframe with features and use original column names
    input_df = pd.DataFrame([features], columns=data.columns)

    # Combines user input features with cleaned dataset
    # This will be useful for the encoding phase
    df = pd.concat([input_df, data],axis=0, ignore_index=True)

    def one_hot_encode(df):
        # get categorical features of df
        cat_feat = df.select_dtypes(exclude = np.number).columns
        for feat in cat_feat:
            df = pd.get_dummies(columns=[feat], data=df, dtype=np.int64)
        input_df = df[:1]  # Selects only the first row (the user input data)
        return input_df

    input_df = one_hot_encode(df)

    # Reads in saved classification model
    load_clf = pickle.load(open('model/stroke_smote_tuned.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(input_df)
    proba = load_clf.predict_proba(input_df)
    prob_df = pd.DataFrame(proba, columns=['No', 'Yes'])
    percent = proba.max() * 100

    # print out prediction
    if st.button('Classify'):
        output = np.array(['No','Yes'])
        # st.write(output[prediction])
        st.markdown('Prediction Probability')
        st.write(prob_df)
        if output[prediction] == 'Yes':
            st.write('Patient at higher risk of Stroke with confidence {:.2f}'.format(percent))
            st.write('Time to Visit your Doctor!')
        else:
            st.write('Patient at lower risk of Stroke with confidence {:.2f}'.format(percent))
            st.write('Time to visit your doctor!')
        st.write('## This is not a replacement for proper medical advice')
    else:
        st.write(' ')


if __name__ == "__main__":
    main()
