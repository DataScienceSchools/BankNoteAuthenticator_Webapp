"""
@author: Bahar GK
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
      page_title='Bank Note Authenticator App',
      page_icon="🧊",
      layout="wide",
      initial_sidebar_state="expanded")

st.title("Bank Note Authenticator APP")
st.write('- Predicting whether a given banknote is authentic given a number of measures taken from a photograph.')
with st.beta_expander("Author"):
    st.markdown("""
    -  Bahar GK
    - [Data Science Schools](https://datascienceschools.github.io/)
    """)
    st.write('---')
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Bank Note Authenticator ML APP </h2>
    </div>
    """

classifier = pickle.load(open("RFClassifier.pkl","rb"))

def predict_note_authentication(variance,skewness,curtosis,entropy):
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    st.subheader('User Input Parameters:')
    st.write('\n\n- Variance: ', variance,
                '\n\n- Skewness: ', skewness,
                '\n\n- Curtosis: ', curtosis,
                '\n\n- Entropy: ', entropy)
    print(prediction)
    return prediction




st.sidebar.header('Upload CSV file')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

def main():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if st.sidebar.button("Predict"):
            df['Prediction'] = classifier.predict(df)
            st.write(df)
    else:
        st.sidebar.header('User Input Parameters')
        st.markdown(html_temp,unsafe_allow_html=True)
        variance = st.sidebar.text_input("Variance","Type Here")
        skewness = st.sidebar.text_input("Skewness","Type Here")
        curtosis = st.sidebar.text_input("Curtosis","Type Here")
        entropy  = st.sidebar.text_input("Entropy","Type Here")
        if st.sidebar.button("Predict"):
            result=""
            result=predict_note_authentication(variance,skewness,curtosis,entropy)
            st.subheader('Prediction:')
            if result[0] == 1:
                st.success(' Banknote Is Authentic (the output is {}'.format(result[0]) + ")")
            else:
                st.success(' Banknote Is Not Authentic (the output is {}'.format(result[0]) + ")")


if __name__=='__main__':
    main()
