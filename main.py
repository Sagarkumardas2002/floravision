import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64

model = load_model('Model/model.h5')
class_dict = np.load("artifacts/class_names.npy")


def predict(image):
    IMG_SIZE = (1, 224, 224, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    return pred

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2670&q=80');
        background-size: cover
        
    }}
   
    </style>
    """,
    unsafe_allow_html=True
    )

contnt = '<p style = "font-family:sans-serif; color: darkyellow; font-size: 20px;">Herbal medicines are preferred in both developing and developed countries as an alternative to " \
         "synthetic drugs mainly because of no side effects. Recognition of these plants by human sight will be " \
         "tedious, time-consuming, and inaccurate.</p> ' \
         '<p style = "font-family:sans-serif; color: black; font-size: 20px;">Applications of image processing and computer vision " \
         "techniques for the identification of the medicinal plants are very crucial as many of them are under " \
         "extinction as per the IUCN records. Hence, the digitization of useful medicinal plants is crucial " \
         "for the conservation of biodiversity.</p>'

if __name__ == '__main__':
    add_bg_from_local("artifacts/Background.jpg")
    new_title = '<p style="font-family:sans-serif; color:darkgreen; font-size: 48px;"><b>Medicinal Leaf Classification</b></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(contnt, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        img = img.resize((300, 300))
        st.image(img)

        if st.button("Predict"):
            pred = predict(img)
            name = class_dict[pred]

            result = '<p style="font-family:sans-serif; color:White; font-size: 20px;">The given image ' \
                        'is '+name+'</p>'
            st.markdown(result, unsafe_allow_html=True)

        


