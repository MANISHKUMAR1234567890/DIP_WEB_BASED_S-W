import streamlit as st  
import cv2
import numpy as np 
from PIL import Image

with subtab[1]:
    st.write("Otsu Segmentation:")
    image = Image.open(uploaded_file).convert("L") 
    img= np.array(image)

     # Apply Otsu's Thresholding
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    otsu_image = Image.fromarray(otsu_thresh)

   
    st.image(otsu_image,caption="Otsu thresholded image", use_container_width=False)
    st.code("""
    import numpy as np
    import cv2
    from PIL import Image
    img=Image.open("path").convert("L")
    image_np=np.array(img)
     # Apply Otsu's Thresholding
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        """,language="python")
    

with subtab[2]:
    st.write("Thresholding:")
    image = Image.open(uploaded_file).convert("L") 
    img= np.array(image)
    _, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
    otsu_image = Image.fromarray(global_thresh)

    st.image(otsu_image,caption="Thresholded image", use_container_width=False)
    st.code("""
    import numpy as np
    import cv2
    from PIL import Image
    img=Image.open("path").convert("L")
    image_np=np.array(img)
     # Apply Thresholding
    _, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
        """,language="python")
    


with subtab[3]:
    st.write("Clustering:")
    image = Image.open(uploaded_file).convert("L")
    img_color = np.array(image)

    # Reshape the image to a 2D array of pixels
    Z = img_color.reshape((-1, 3))
    Z = np.float32(Z)

    # Define K-Means criteria and apply it
    K = st.slider("Select number of clusters (K)", 2, 10, 4)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and reshape to original image shape
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape((img_color.shape))
    st.image(image, caption="Clustered Image",use_container_width=False)
    st.code("""
            import numpy as np
            import cv2
            from PIL import Image
            img=Image.open("path").convert("L")
             img_color = np.array(image)

            # Reshape the image to a 2D array of pixels
             Z = img_color.reshape((-1, 3))
             Z = np.float32(Z)

             # Define K-Means criteria and apply it
             K = Define K-Means criteria
             criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
             _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

             # Convert back to uint8 and reshape to original image shape
            centers = np.uint8(centers)
            segmented_img = centers[labels.flatten()]
            segmented_img = segmented_img.reshape((img_color.shape))""",language="python")

            
