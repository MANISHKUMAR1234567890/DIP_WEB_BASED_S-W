import cv2
import numpy as np
from PIL import Image
import streamlit as st



with subtab[1]:
    subsubtab = st.tabs(["Select",'Mean filter','Gaussian Filter','Median Filter'])
    with subsubtab[1]:
        st.write("Apply Mean Filter")
        img=Image.open(uploaded_file)
        img_array = np.array(img, dtype=np.uint8)# Convert to NumPy array

    # Define mean filter kernel
        mean_kernel = np.ones((3, 3), np.float32) / 9

    # Apply mean filter
        mean_filtered = cv2.filter2D(img_array, -1, mean_kernel)
        
        st.image(mean_filtered,caption="Mean filtered image",use_container_width=False)
        st.code("""
                import cv2
                import numpy as np
                from PIL import Image
                img=Image.open("path")
                imag_array=np.array(img, dtype=np.uint8)
                mean_kernel = np.ones((3,3), np.float32) / 9
                mean_filtered = cv2.filter2D(img_array, -1, mean_kernel)

""",language='python')
    with subsubtab[2]:
        st.write("Apply Gaussian Filter")
        img=Image.open(uploaded_file)
        img_array=np.array(img, dtype=np.uint8)
        gausian_filter=cv2.GaussianBlur(img_array, (3,3), 1)
        st.image(gausian_filter,caption="Gaussian filtered image",use_container_width=False)
        st.code("""
                import cv2
                import numpy as np
                from PIL import Image
                img=Image.open("path")
                imag_array=np.array(img, dtype=np.uint8)
                mean_filtered=cv2.GaussianBlur(imag_array, kernel_size, 1)

""",language='python')
    with subsubtab[3]:
        st.write("Apply Median Filter")
        img=Image.open(uploaded_file)
        img_array=np.array(img, dtype=np.uint8)
        median_filtered = cv2.medianBlur(img_array, 3)
        st.image(median_filtered,caption="Median filtered image",use_container_width=False)
        st.code("""
                import cv2
                import numpy as np
                from PIL import Image
                img=Image.open("path")
                imag_array=np.array(img, dtype=np.uint8)
                median_filtered = cv2.medianBlur(imag_array, kernel_size)

""",language='python')
        

with subtab[2]:
    subsubtab = st.tabs(["Select",'Laplacian filter', 'Sobel Filter','Unsharp mask'])
    with subsubtab[1]:
        st.write("Apply Laplacian Filter")
        img=Image.open(uploaded_file)
        img_array=np.array(img)
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)  # Compute Laplacian (float64 output)
        laplacian_abs = cv2.convertScaleAbs(laplacian)  # Convert to uint8
        laplacian_sharpened = cv2.subtract(img_array, laplacian_abs)  # Sharpening by subtracting edges
        st.image(laplacian_sharpened,caption="Laplacian fitered Image",use_container_width=False)
        st.code("""
                import numpy as np
                import cv2
                from PIL import Image
                img=Image.open("path")
                img_array=np.array(img)
                laplacian = cv2.Laplacian(img_array, cv2.CV_64F)  # Compute Laplacian (float64 output)
                laplacian_abs = cv2.convertScaleAbs(laplacian)  # Convert to uint8
                laplacian_sharpened = cv2.subtract(img_array, laplacian_abs)
""",language='python')
# So
    with subsubtab[2]:
        st.write("Apply Sobel Filter")
        img=Image.open(uploaded_file)
        img_array=np.array(img)
        sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.convertScaleAbs(cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0))

        st.image(sobel_combined,caption="Sobel fitered Image",use_container_width=False)
        st.code("""
                import numpy as np
                import cv2
                from PIL import Image
                img=Image.open("path")
                img_array=np.array(img)
                sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=5)
                sobel_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=5)
                sobel_combined = cv2.convertScaleAbs(cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0))

""",language='python')
    with subsubtab[3]:
        st.write("Apply Unsharp Mask")
        img=Image.open(uploaded_file)
        img_array=np.array(img)
        gaussian_blurred = cv2.GaussianBlur(img_array, (3,3), 1)  # Blur image
        unsharp_masked = cv2.addWeighted(img_array, 1.5, gaussian_blurred, -0.5, 0)  # Sharpen

        st.image(unsharp_masked,caption="Unsharp masked Image",use_container_width=False)
        st.code("""
                import numpy as np
                import cv2
                from PIL import Image
                img=Image.open("path")
                img_array=np.array(img)
                gaussian_blurred = cv2.GaussianBlur(img_array, (3,3), 1)  # Blur image
                unsharp_masked = cv2.addWeighted(img_array, 1.5, gaussian_blurred, -0.5, 0)  # Sharpen

""",language='python')
