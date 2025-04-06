import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="DIP Assistant",layout="wide",page_icon="🤖")



st.title("Interactive Digital Image Processing Software")


import streamlit as st
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("image.png")


st.markdown("""
<style>
  
    img {
 
        max-width: 1000px !important;
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

tab=st.tabs(["Home", "          ","Learn About DIP","    ","Practical Demostration","   "  ,"About"])

with tab[0]:

    st.markdown("""
    Welcome to the **Digital Image Processing (DIP)** Web App!  
    This tool is designed to provide users with hands-on experience in applying various image processing techniques in a simple and interactive way.

Whether you're a student, researcher, or developer, this app will help you visualize and understand the magic of image transformation.
""")


     



with tab[2]:
    exec(open("chatbot.py").read())
    


with tab[4]:

    
  

    st.title(" Digital Image Processing Demostration")


    import streamlit as st
    import base64

    def add_bg_from_local(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    add_bg_from_local("image.png")


    st.markdown("""
<style>
  
    img {
 
        max-width: 1000px !important;
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

    tabs = st.tabs(["Original","Image Preprocessing","Pixel Processing",
"Spatial filter","Frequency filter","Noise Model","Morphology",
"Edge Detection","Compression","Segmentation","Discontinuity Detection"])

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
            st.image(img_resized, caption="Uploaded Image", use_column_width=True)
            st.code("print('Original Image Loaded')", language="python")


        with tabs[1]:
            subtab = st.tabs(["Select","Conversion","Normalization","Croping","Resizing"])
            exec(open("PREPROCESSING.py").read())

        with tabs[2]:
            subtab = st.tabs(["Select",'Brightness Adjustment','Contrast Adjustment','Histogram Equilization'])
            exec(open("pixel_processing.py").read())







        with tabs[3]:
            subtab = st.tabs(["Select",'Smothening','Sharpening'])
            exec(open("spatial_filter.py").read())
        
        
        with tabs[4]:
            subtab=st.tabs(["Select","Low Pass Filter","High Pass Filter"])
            exec(open("frequency_filter.py").read())

        with tabs[5]:
            subtab=st.tabs(["Select","Gaussian Noise","Rayleigh Noise","Erlang Noise","Salt and Pepper Noise"])
            exec(open("Noise models.py").read())


        with tabs[6]:
            exec(open("morphology.py").read())

        with tabs[7]:
            subtab=st.tabs(["Select","Sobel Edge Detection","Prewitt Edge Detection","Canny Edge Detection"])
            exec(open("edge.py").read())

        with tabs[8]:
            subtab=st.tabs(["Select","Lossless compression","JPEG compression"])
            exec(open("compression.py").read())

        with tabs[9]:
            subtab=st.tabs(["Select","Otsu","Thresholding","Clustering"])
            exec(open("segmentation.py").read())

        with tabs[10]:
            exec(open("discontinuity_detection.py").read())

 





  
with tab[6]:
    st.title("About This Software")
    st.markdown("""
Welcome to the **Digital Image Processing (DIP) Web Application**! 🧠📷

This software is designed for **students, researchers, and beginners** to **learn and explore the concepts of digital image processing** through practical demonstrations.

---

### 🔍 What You Can Do with This App:

- 📤 **Upload an image** from your device
- 🛠️ **Apply various image processing operations** such as:
    - Grayscale Conversion
    - Thresholding (Otsu, Global)
    - Edge Detection
    - Clustering (K-Means)
    - Discontinuity Detection
    - And more...
- 🧾 **View the Python code** corresponding to each operation
- 🎓 Enhance your **understanding of image processing techniques**

---

### 🎯 Purpose

This app serves as an **interactive learning platform** to bridge the gap between theory and implementation in Digital Image Processing.

Whether you're exploring DIP for the first time or revising key concepts, this tool will help you **visualize the effects of different algorithms** on images while giving you the **actual Python code** for your reference or academic use.

---

Thank you for using this app! 😊
""")




st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .custom-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #262730;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            font-family: Arial, sans-serif;
            z-index: 100;
        }

        .custom-footer a {
            color: #FFD700; /* Gold color */
            text-decoration: none;
            margin: 0 10px;
        }

        .custom-footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="custom-footer">
        © 2025 Manish Kumar  |  <a href="">manishk97253@gmail.com</a> |
         <a href="https://www.linkedin.com/in/manish-kumar-79b2a8276/">Linkedin</a>
    </div>
""", unsafe_allow_html=True)
