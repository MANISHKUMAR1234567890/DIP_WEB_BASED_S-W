import cv2
import numpy as np
from PIL import Image
import streamlit as st


with subtab[1]:
    st.write("Adjust Image Brightness")

    image = Image.open(uploaded_file)
    image_np = np.array(image)
    brightness = st.slider("Adjust Brightness", -100, 100, 0)
    bright_image = cv2.convertScaleAbs(image_np, alpha=1, beta=brightness)
    st.image(bright_image,caption="Brighten image",use_column_width=False)
    st.code("""
            import numpy as np
            import cv2
            from PIL import Image
            img=Image.open("path").conver("L")
            img_array=np.array(img)
            brigntness="Set range"
            bright_image=cv2.convertScaleAbs(img_array, alpha=1, beta=brightness)
""",language='python')
with subtab[2]:
    st.write("Adjust Image Contrast")
    image = Image.open(uploaded_file).convert("L") 
    image_np = np.array(image)
    r_min, r_max = np.min(image_np), np.max(image_np)
    s_min = st.slider("Select Min Output Intensity", 0, 255, 0)
    s_max = st.slider("Select Max Output Intensity", 0, 255, 255)
    contrast_stretched = ((image_np - r_min) / (r_max - r_min)) * (s_max - s_min) + s_min
    contrast_stretched = np.clip(contrast_stretched, s_min, s_max).astype(np.uint8)
    st.image(contrast_stretched,caption="Contrast Stretched",use_column_width=False)
    st.code("""
             import numpy as np
            import cv2
            from PIL import Image
            img=Image.open("path").conver("L")
            img_array=np.array(img)
            r_min, r_max = np.min(img_array), np.max(img_array)
            smin="set minimum output range"
            smax="set maximum output range"
            contrast_stretched = ((image_np - r_min) / (r_max - r_min)) * (s_max - s_min) + s_min
            contrast_stretched = np.clip(contrast_stretched, s_min, s_max).astype(np.uint8)
            """,language="python")


with subtab[3]:
    st.write("Apply Histogram Equalization")
    image = Image.open(uploaded_file).convert("L") 
    image_np = np.array(image)
    equalized_image = cv2.equalizeHist(image_np)
    st.image(equalized_image,caption="Histogram equilized image",use_column_width=False)
    st.code("""
            import numpy as np
            import cv2
            from PIL import Image
            img=Image.open("path").conver("L")
            img_array=np.array(img)
            equilized_histogram=cv2.equilizeHist(img_array)
            """, language='python')
