import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

model = tf.keras.models.load_model('inceptV3Bug8571-x.h5')

classes = ['Bladlöss', 'Sköldlöss', 'Spinnkvalster', 'Ullöss']

img_size = 200

def prediction(img):
    image = Image.open(img)
    resized = image.resize((img_size, img_size))
    npImg = np.array(resized)
    rescaled = npImg / 255
    imgage = rescaled.reshape(1, img_size, img_size, 3)

    probs = model.predict(imgage)

    st.write(f'Prediction : {classes[np.argmax(probs)]}')
    st.write('')
    st.write('(Resized image)')
    st.image(resized)