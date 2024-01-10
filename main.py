import numpy as np
import streamlit as st


with st.sidebar:
    st.markdown("#### **INFORMATIONS NEEDED**")
    with st.form("questions"):
        age = st.number_input("Please type your age",
                            min_value=18,
                            max_value=120)
        sex = st.radio("Please select your gender",
                       ["female", "male"],
                       horizontal=True)
        weight = st.number_input("Please type your age (in kg)",
                                 min_value=20,
                                 max_value=220,
                                 value=60)
        height = st.slider("Please select your height (in cm)",
                           min_value=0.60,
                           max_value=2.30,
                           value=1.60,
                           step=0.01)
        children = st.number_input("Please type the number of children you have",
                                   min_value=0,
                                   value=0)
        smoker = st.radio("Please specify if you smoke or not",
                          ["yes", "no"],
                          horizontal=True)
        region = st.radio("Please select the region you live in",
                          ["northeast", "northwest", "souteast", "southwest"])
        submit_btn = st.form_submit_button("Submit")


smoker_display = "smoker" if smoker == "yes" else "non smoker"
st.title("PolicyPriceAI")
st.divider()
st.header("Let's evaluate your annual insurance charges")
st.markdown("#### **Please EXPAND the sidebar on the left**")
st.divider()
st.markdown(f"""
            ### You specified the following informations:
            - **{age}** years old,
            - **{sex}**, 
            - weight: **{weight}** kg , height: **{height:.2f}** m 
            - **{smoker_display}**
            - living in **{region}**
            """)
st.divider()
st.caption("With these informations, here are your annual charges for insurance:")

bmi = weight / (height ** 2)


X_user = np.array([age, sex, bmi, children, smoker, region])