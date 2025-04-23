import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO("bee.pt")

st.title("🐝 Queen Bee Detection using YOLOv8")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with PIL and convert to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO prediction
    results = model(image_bgr)

    # Visualize results
    annotated_frame = results[0].plot()

    # Convert BGR to RGB for display
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Resize image to 40%
    scale_percent = 40
    width = int(annotated_frame_rgb.shape[1] * scale_percent / 100)
    height = int(annotated_frame_rgb.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(annotated_frame_rgb, (width, height), interpolation=cv2.INTER_AREA)

    # Display the annotated image
    st.image(resized_frame, caption="Detected Image", use_column_width=True)
