import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pytesseract
import cv2
import numpy as np

# Load your custom trained YOLOv5 model
from roboflow import Roboflow
rf = Roboflow(api_key="J0F9PiAO9F3P0I5g5viY")
project = rf.workspace().project("amrita-id-card-detection")
model = project.version(3).model


# Set page title and favicon
st.set_page_config(page_title="Amrita ID Card Detector", page_icon="🔍")

# Display sidebar
st.sidebar.title("About")
st.sidebar.info("This app is a demonstration of the Amrita School of Engineering ID Card Detector. The model is trained using a custom YOLOv5 model.")

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

    # Progress bar
    with st.spinner('Processing...'):
        # Call the predict method on your model
        results = model.predict("temp.jpg", confidence=28, overlap=30).save("prediction.jpg")

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
    img = cv2.imread('prediction.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get image with only black and white
    _, binary_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the image
    inverted_binary_img = cv2.bitwise_not(binary_img)

    # Save and show processed image
    cv2.imwrite('processed.jpg', inverted_binary_img)
    st.image('processed.jpg', caption='Processed Image.', use_column_width=True)

    # Apply OCR to the processed image
    custom_config = r'--oem 3 --psm 6'  # Set the OEM and PSM parameters
    text = pytesseract.image_to_string(inverted_binary_img, config=custom_config, lang = 'eng')
   



    # Print the result
    #print(result[0]['sequence'])
    # Display the extracted text
    st.write("Extracted Text: ")
    st.write(text)

else:
    st.error("Please upload a JPG file.")

# Footer
st.markdown("---")