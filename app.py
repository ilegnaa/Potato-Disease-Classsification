import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model('final_model.h5')

# Define disease classes
classes = ['Early Blight', 'Late Blight', 'Healthy']

def preprocess_img(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def classify_image(img):
    img = preprocess_img(img)
    prediction = model.predict(img)
    return prediction

def main():
    st.title('Potato Leaf Disease Classifier')
    st.write('Upload an image of a potato leaf and we will classify it.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        prediction = classify_image(img)
        class_idx = np.argmax(prediction)
        result = classes[class_idx]
        st.write('Prediction:', result)

if __name__ == '__main__':
    main()
