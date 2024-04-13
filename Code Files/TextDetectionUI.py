import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pytesseract
import cv2
import numpy as np

# Load your custom trained YOLOv5 models
from roboflow import Roboflow
rf = Roboflow(api_key="J0F9PiAO9F3P0I5g5viY")

# Load models
project = rf.workspace().project("amrita-id-card-detection")
model = project.version(3).model
project2 = rf.workspace().project("id-text-detection")
model2 = project2.version(2).model

# Set page title and favicon
st.set_page_config(page_title="Amrita ID Card Detector", page_icon="🔍")

# Display sidebar
st.sidebar.title("About")
st.sidebar.info("This app is a demonstration of the Amrita School of Engineering ID Card Detector. The models are trained using custom YOLOv5 models.")

st.title("Amrita School of Engineering ID Card Detector")
st.write("Upload an image to get started.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    st.write("Detecting...")
    # Save the uploaded image to a temporary file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Call the predict method on your model
    with st.spinner('Processing...'):
        results = model.predict("temp.jpg", confidence=40, overlap=30).save("prediction.jpg")

    # Display the results
    st.success("Detection complete!")
    st.image('prediction.jpg', caption='Processed Image.', use_column_width=True)

    # Getting the bounding box coordinates
    bbox_data = model.predict("temp.jpg", confidence=40, overlap=30).json()
    bbox = bbox_data['predictions'][0]

    # Open image using OpenCV
    img = cv2.imread('temp.jpg')

    # Convert coordinates to integers
    x = int(bbox['x'])
    y = int(bbox['y'])
    width = int(bbox['width'])
    height = int(bbox['height'])

    # Calculate top left (x1, y1) and bottom right (x2, y2) coordinates
    x1, y1 = max(0, x - width // 2), max(0, y - height // 2)
    x2, y2 = min(img.shape[1] - 1, x + width // 2), min(img.shape[0] - 1, y + height // 2)

    # Crop image
    cropped_img = img[y1:y2, x1:x2]
    cv2.imwrite('temp_cropped.jpg', cropped_img)
    
    # Running second model on cropped image
    with st.spinner('Processing cropped image...'):
        results2 = model2.predict("temp_cropped.jpg", confidence=28, overlap=30).save("prediction_cropped.jpg")

    # Display the results
    st.success("Detection on cropped image complete!")
    st.image('prediction_cropped.jpg', caption='Processed Cropped Image.', use_column_width=True)

    # Getting the bounding box coordinates
    bbox_data2 = model2.predict("temp_cropped.jpg", confidence=40, overlap=30).json()
    bboxes = bbox_data2['predictions']

    # Extract text for each bounding box
    tags = ['Name', 'ID NO.', 'Branch', 'Validity']  # Assuming these are the tags you mentioned
    for i, bbox in enumerate(bboxes):
        # Convert coordinates to integers
        x = int(bbox['x'])
        y = int(bbox['y'])
        width = int(bbox['width'])
        height = int(bbox['height'])

        # Calculate top left (x1, y1) and bottom right (x2, y2) coordinates
        x1, y1 = max(0, x - width // 2), max(0, y - height // 2)
        x2, y2 = min(cropped_img.shape[1] - 1, x + width // 2), min(cropped_img.shape[0] - 1, y + height // 2)

        # Crop image
        cropped_bbox = cropped_img[y1:y2, x1:x2]
        cv2.imwrite('temp_bbox.jpg', cropped_bbox)

        # Display the cropped image
        st.image('temp_bbox.jpg', caption=f'Cropped Image for {tags[i]}', use_column_width=True)

        # Apply OCR to the processed image
        custom_config = r'--oem 3 --psm 11'  # Set the OEM and PSM parameters
        text = pytesseract.image_to_string(cropped_bbox, config=custom_config, lang = 'eng')

        # Display the extracted text
        st.write(f"Extracted Text for {tags[i]}: ")
        st.write(text)

else:
    st.error("Please upload a JPG file.")

# Footer
st.markdown("---")
