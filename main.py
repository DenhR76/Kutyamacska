import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

st.title("Állatfelismerő alkalmazás")
st.write("Tölts fel egy képet, hogy felismerjük rajta az állatokat!")


uploaded_file = st.file_uploader("Tölts fel egy képet", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    
    results = model(image)

    
    annotated_image = results[0].plot()  
    st.image(annotated_image, channels="BGR", caption="Felismert állatok")

    
    st.write("Felismerések:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  
        confidence = box.conf[0]  
        label = model.names[cls_id] 
        st.write(f"- {label}: {confidence * 100:.2f}%")

