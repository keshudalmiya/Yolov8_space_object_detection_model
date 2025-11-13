# streamlit_app.py — quick UI to upload an image and run detection
import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np


MODEL = None


@st.cache_resource
def load_model(weights='runs/train/exp/weights/best.pt'):
return YOLO(weights)


st.title('Space Station Safety Detection — Demo')
uploaded_file = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
if uploaded_file is not None:
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())
img = cv2.imread(tfile.name)
st.image(img[:,:,::-1], caption='Input')
if st.button('Detect'):
model = load_model()
results = model.predict(source=tfile.name, conf=0.25)
res_img = results[0].plot()
st.image(res_img)
