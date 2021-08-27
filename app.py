import streamlit as st
import predict as pred

st.title('Pest classification with tensorflow')
st.header('This is a pest classifier trained in tensorflow with 85.71 % accuracy')
st.subheader('')
st.text('Upload a picture')

uploaded_file = st.file_uploader('upload a picture', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:

    st.write('(Orginal image)')
    st.image(uploaded_file)
    pred.prediction(uploaded_file)
