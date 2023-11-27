import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

from PIL import Image


# Title/ Text

html_temp = """
<div style="background-color:Red;padding:10px">
<h2 style="color:white;text-align:center;">Predicting the Car Price</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.text("Are you curious aboout the car prices?\nOur model will help you make prediction based on certine features of the car.")

# Image
img = Image.open("car.jpg")
st.image(img,width = 200)


# Sidebar
st.sidebar.title("Assign the features values")

# Sidebar inputs

car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))
age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))


ds13_model=pickle.load(open("rf_model_new","rb"))
ds13_transformer = pickle.load(open('transformer', 'rb'))

#Display selected features in a table with modified colors
selected_features = pd.DataFrame({
    'Feature': ['Car Make Model','Age', 'hp_kW','Kilometers', 'Geaning_Type'],
    'Selected Value': [car_model, age, hp, km, gearing_type]
})

# Style the table with custom colors
st.subheader("Summary table of the selected values")
st.table(selected_features.style.set_table_styles([{
    'selector': 'th',
    'props': [('background-color', 'red'), ('color', 'white')]
}, {
    'selector': 'td',
    'props': [('background-color', 'white'), ('color', 'black')]
}]))



# Prepare input features for prediction
input_data = pd.DataFrame({
    'age': [age],
    'hp_kW': [hp],
    'make_model': [car_model],
    'Gearing_Type': [gearing_type],
    'km': [km]
})

df2 = ds13_transformer.transform(input_data)

st.subheader("Press predict to see the results")

if st.button("Predict"):
    prediction = ds13_model.predict(df2)
    st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))
