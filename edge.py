import numpy as np
import cv2 
import streamlit as st  
from PIL import Image



with subtab[1]:
    st.write("Sobel Edge Detection: ")
    image = Image.open(uploaded_file).convert("L") 
    image_np = np.array(image)
    sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
    # Compute magnitude of gradients
    sobel_edge = cv2.magnitude(sobel_x, sobel_y)
    # Normalize and convert to uint8 for display
    sobel_edge = np.uint8(255 * sobel_edge / np.max(sobel_edge))
    st.image(sobel_edge,caption="Edges detected",use_column_width=False)
    st.code("""
            import numpy as np
            import cv2
            from PIL import Image
            img=Image.open("path").convert("L")
            image_np=np.array(img)
            sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
            # Compute magnitude of gradients
            sobel_edge = cv2.magnitude(sobel_x, sobel_y)
            # Normalize and convert to uint8 for display
            sobel_edge = np.uint8(255 * sobel_edge / np.max(sobel_edge))""",language="python")
    


with subtab[2]:
    st.write("Prewitt Edge Detection: ")
    image = Image.open(uploaded_file).convert("L") 
    image_np = np.array(image)
    # Apply Prewitt Edge Detection
    prewitt_kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_kernel_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_x = cv2.filter2D(image_np, cv2.CV_32F, prewitt_kernel_x)
    prewitt_y = cv2.filter2D(image_np, cv2.CV_32F, prewitt_kernel_y)
    prewitt_edge = cv2.magnitude(prewitt_x, prewitt_y)
    prewitt_edge = np.uint8(255 * prewitt_edge / np.max(prewitt_edge))
    st.image(prewitt_edge,caption="Edges Detected",use_column_width=False)
    st.code("""
            import numpy as np
            import cv2
            from PIL import Image
            img=Image.open("path").convert("L")
            image_np=np.array(img)
             # Apply Prewitt Edge Detection
            prewitt_kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            prewitt_kernel_y = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            prewitt_x = cv2.filter2D(image, cv2.CV_32F, prewitt_kernel_x)
            prewitt_y = cv2.filter2D(image, cv2.CV_32F, prewitt_kernel_y)
            prewitt_edge = cv2.magnitude(prewitt_x, prewitt_y)
            prewitt_edge = np.uint8(255 * prewitt_edge / np.max(prewitt_edge))
            """,language="python")
    



with subtab[3]:
    st.write("Canny Edge Detection:")
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)
    # Apply Canny Edge Detection
    canny_edge = cv2.Canny(image_np, 100, 200)  
    canny_edge_rgb = cv2.cvtColor(canny_edge, cv2.COLOR_GRAY2RGB)  # Convert to RGB for Streamlit
    # Display the image
    st.image(canny_edge_rgb, caption="Edges detected", use_column_width=False)
    st.code("""
            import numpy as np
            import cv2
            from PIL import Image
            img=Image.open("path").convert("L")
            img_array=np.array(img)
            # Apply Canny Edge Detection
            canny_edge = cv2.Canny(image_np, 100, 200)  
            canny_edge_rgb = cv2.cvtColor(canny_edge, cv2.COLOR_GRAY2RGB)  
            """,language="python")
    


