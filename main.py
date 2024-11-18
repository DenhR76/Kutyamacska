import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO modell betöltése (használj egy előre betanított vagy saját modellt)
model = YOLO("yolov8n.pt")  # Helyettesítsd saját "best.pt" modelleddel, ha van

# Alkalmazás címe
st.title("Állatfelismerő alkalmazás")
st.write("Tölts fel egy képet, hogy felismerjük rajta az állatokat!")

# Kép feltöltése
uploaded_file = st.file_uploader("Tölts fel egy képet", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Kép beolvasása byte formátumból
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # YOLO felismerés
    results = model(image)

    # Eredmények megjelenítése
    annotated_image = results[0].plot()  # Az eredményekkel ellátott kép
    st.image(annotated_image, channels="BGR", caption="Felismert állatok")

    # Eredmények listázása
    st.write("Felismerések:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  # Osztály azonosítója
        confidence = box.conf[0]  # Valószínűség
        label = model.names[cls_id]  # Osztály neve
        st.write(f"- {label}: {confidence:.2f}")
