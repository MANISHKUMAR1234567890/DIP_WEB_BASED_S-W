import cv2
import numpy as np
import streamlit as st
from PIL import Image


def apply_morphology(image, operation, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Structuring element

    if operation == "Erosion":
        return cv2.erode(image, kernel, iterations=1)
    elif operation == "Dilation":
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == "Opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def generate_code(operation):
    if operation == "Erosion":
        code_template = f"""
import cv2
import numpy as np
image = cv2.imread("your_image.jpg", 0)
kernel = select kernel size 
result = cv2.erode(image, kernel, iterations=1)
"""
    elif operation == "Dilation":
        code_template = f"""
import cv2
import numpy as np
image = cv2.imread("your_image.jpg", 0)
kernel = select kernel size
result = cv2.dilate(image, kernel, iterations=1)
"""
    elif operation == "Opening":
        code_template = f"""
import cv2
import numpy as np
image = cv2.imread("your_image.jpg", 0)
kernel = select kernel size
result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
"""
    elif operation == "Closing":
        code_template = f"""
import cv2
import numpy as np
image = cv2.imread("your_image.jpg", 0)
kernel = select kernel size 
result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
""" 
    return code_template

img=Image.open(uploaded_file).convert("L")
img_array=np.array(img)
operation = st.selectbox("Select Morphological Operation", ["Erosion", "Dilation", "Opening", "Closing"])
kernel_size = st.slider("Select Kernel Size", 3, 15, 3, step=2)  
result = apply_morphology(img_array, operation, kernel_size)
st.image(result, caption="Morphology operation applied", use_container_width=False)
st.code(generate_code(operation), language="python")
