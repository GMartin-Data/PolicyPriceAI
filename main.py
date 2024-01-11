import joblib
import numpy as np
import streamlit as st


# Loading (and caching) the model
@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# # Uncomment when "model.pkl" is generated
# model = load_model("model.pkl")

## Gathering informations
with st.sidebar:
    st.markdown("""<img src='https://i.ibb.co/SwN5Mjy/Logo-shield-resized.png'>
                powered by <strong>PolicyPriceAI</strong>
                """, unsafe_allow_html=True)
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

## Disclaimer, and input's double check
smoker_display = "smoker" if smoker == "yes" else "non smoker"
st.title("PolicyPriceAI")
st.divider()
st.markdown("### Let's evaluate your annual insurance charges")
st.markdown("#### **Please EXPAND the sidebar on the left**")
st.divider()
if submit_btn:
    st.markdown(f"""
                ### You specified the following informations:
                - **{age}** years old, **{sex}**, 
                - weight: **{weight}** kg , height: **{height:.2f}** m 
                - **{smoker_display}**
                - living in **{region}**
                """)
    st.divider()
    st.markdown("#### With these informations, here are your annual charges for insurance:")


    ## Performing the prediction
    bmi = weight / (height ** 2)
    # Reshaping because we need a 2D-array as pipeline input
    X_user = np.array([age, sex, bmi, children, smoker, region]).reshape(-1, 1)

    # # The output should be a 1D-array with one value
    # log_fees = model.predict(X_user)[0]
    # Transform fees to retrieve original target
    # fees = np.exp(log_fees) - 1
    fees = 12345.67890

    st.markdown(f"""
    #### **{fees:.2f}** $
                """)