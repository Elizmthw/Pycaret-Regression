from pycaret.regression import load_model,predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('FinalModel')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions=predictions_df['Label'][0]
    return predictions


def run():
    from PIL import Image
    image = Image.open('image1.jpg')
    image_office = Image.open('image2.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox('How would you like to predict?',('Online','Batch'))

    st.sidebar.info('This app is created to predict the salary of a person based on his/her years of experience')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)


    st.title('Salary Prediction')

    if add_selectbox == 'Online':
        options=list(np.arange(0,100,1))
        companyId = st.selectbox('companyId',options,format_func=lambda x: "COMP"+str(x))
        jobType = st.selectbox('jobType',['CFO','CTO','CEO','MANAGER','VICE_PRESIDENT','SENIOR','JUNIOR','JANITOR'])
        degree = st.selectbox('degree',['NONE','HIGH_SCHOOL','DOCTORAL','MASTERS','BACHELORS'])
        major = st.selectbox('major',['NONE','BUSINESS','LITERATURE','ENGINEERING','PHYSICS','BIOLOGY','CHEMISTRY','COMPSCI','MATH'])
        industry = st.selectbox('industry',['FINANCE','HEALTH','OIL','SERVICE','WEB','AUTO','EDUCATION'])
        experience = st.number_input('experience', min_value=0, max_value=50,value=1)
        miles = st.number_input('miles', min_value=0, max_value=100,value=1)
        
        output = ""

        input_dict ={'companyId': companyId,
                     'jobType': jobType,
                     'degree':  degree,
                     'major': major,
                     'industry': industry,
                     'experience': experience,
                     'miles': miles}
        
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
        
        
   
def main():
    run()



if __name__ == "__main__":
    main()
