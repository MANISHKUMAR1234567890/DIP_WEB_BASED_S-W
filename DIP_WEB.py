import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

st.title("Interactive Digital Image Processing Software")

st.markdown("""
<style>
  
    img {
 
        max-width: 1000px !important;
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Original","Image Preprocessing","Pixel Processing",
"Spatial filter","Frequency filter","Geometric Transformation","Morphology",
"Edge Detection","Compression","Segmentation","Image Restoration"])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

def resize_image(img, scale=0.2):  # Reduced scale to fit on screen
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height))

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_resized = resize_image(img_array, scale=0.2)
    
    with tabs[0]:
        st.header("Original Image")
        st.image(img_resized, caption="Uploaded Image", use_column_width=False)
        st.code("print('Original Image Loaded')", language="python")


        with tabs[1]:
            subtab = st.tabs(["Select","Conversion","Normalization","Croping","Resizing"])
            exec(open("PREPROCESSING.py").read())

    with tabs[2]:
        subtab = st.tabs(["Select",'Brightness Adjustment','Contrast Adjustment','Histogram Equilization'])
        with subtab[1]:
            st.write("Adjust Image Brightness")
        with subtab[2]:
            st.write("Adjust Image Contrast")
        with subtab[3]:
            st.write("Apply Histogram Equalization")

    with tabs[3]:
        subtab = st.tabs(["Select",'Smothening','Sharpening'])
        with subtab[1]:
            subsubtab = st.tabs(["Select",'Mean filter','Gaussian Filter','Median Filter'])
            with subsubtab[1]:
                st.write("Apply Mean Filter")
            with subsubtab[2]:
                st.write("Apply Gaussian Filter")
            with subsubtab[3]:
                st.write("Apply Median Filter")
        
        with subtab[2]:
            subsubtab = st.tabs(["Select",'Laplacian filter', 'Sobel Filter','Unsharp mask'])
            with subsubtab[1]:
                st.write("Apply Laplacian Filter")
            with subsubtab[2]:
                st.write("Apply Sobel Filter")
            with subsubtab[3]:
                st.write("Apply Unsharp Mask")
