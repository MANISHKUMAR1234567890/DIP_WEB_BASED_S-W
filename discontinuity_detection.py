import streamlit as st 
import cv2
import numpy as np
from PIL import Image
st.write("Discontinuity Detection:")

image = Image.open(uploaded_file).convert("L")
img = np.array(image)

option = st.selectbox("Choose Discontinuity Type", ["Point Detection", "Line Detection", "Edge Detection"])

if option == "Point Detection":
        # Define a Laplacian kernel for point detection
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])
        point_detected = cv2.filter2D(img, -1, kernel)
        st.image(point_detected, caption="Point Discontinuity Detection", use_container_width=False)

        st.code("""
        import cv2
        import numpy as np
        from PIL import Image

        image = Image.open("path").convert("L")
        img = np.array(image)

        # Point Detection using Laplacian kernel
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])
        point_detected = cv2.filter2D(img, -1, kernel)
        """, language="python")

elif option == "Line Detection":
        direction = st.selectbox("Select Line Direction", ["Horizontal", "Vertical", "45 Degree", "135 Degree"])

        if direction == "Horizontal":
            kernel = np.array([[-1, -1, -1],
                               [2, 2, 2],
                               [-1, -1, -1]])
        elif direction == "Vertical":
            kernel = np.array([[-1, 2, -1],
                               [-1, 2, -1],
                               [-1, 2, -1]])
        elif direction == "45 Degree":
            kernel = np.array([[-1, -1, 2],
                               [-1, 2, -1],
                               [2, -1, -1]])
        else:  # 135 Degree
            kernel = np.array([[2, -1, -1],
                               [-1, 2, -1],
                               [-1, -1, 2]])

        line_detected = cv2.filter2D(img, -1, kernel)
        st.image(line_detected, caption=f"Line Discontinuity Detection ({direction})", use_container_width=False)

        st.code(f"""
        import cv2
        import numpy as np
        from PIL import Image

        image = Image.open("path").convert("L")
        img = np.array(image)

        # Line Detection Kernel - {direction}
        kernel = np.array({kernel.tolist()})
        line_detected = cv2.filter2D(img, -1, kernel)
        """, language="python")

elif option == "Edge Detection":
    method = st.selectbox("Edge Detection Method", ["Sobel", "Prewitt", "Canny"])
    scale = st.slider("Edge Scale Factor", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

    if method == "Sobel":
        edge_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        edge_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge = cv2.magnitude(edge_x, edge_y)
        edge = cv2.convertScaleAbs(edge, alpha=scale)

        code_str = """
import cv2
import numpy as np
from PIL import Image

image = Image.open("path").convert("L")
img = np.array(image)

# Edge Detection using Sobel
edge_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
edge_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
edge = cv2.magnitude(edge_x, edge_y)
edge = cv2.convertScaleAbs(edge, alpha=1.0)
"""
    elif method == "Prewitt":
        kernelx = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])
        kernely = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])
        edge_x = cv2.filter2D(img, -1, kernelx)
        edge_y = cv2.filter2D(img, -1, kernely)
        edge = edge_x + edge_y
        edge = cv2.convertScaleAbs(edge, alpha=scale)

        code_str = """
import cv2
import numpy as np
from PIL import Image

image = Image.open("path").convert("L")
img = np.array(image)

# Edge Detection using Prewitt
kernelx = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])
kernely = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])
edge_x = cv2.filter2D(img, -1, kernelx)
edge_y = cv2.filter2D(img, -1, kernely)
edge = edge_x + edge_y
edge = cv2.convertScaleAbs(edge, alpha=1.0)
"""
    else:  # Canny
        edge = cv2.Canny(img, 50, 150)
        code_str = """
import cv2
import numpy as np
from PIL import Image

image = Image.open("path").convert("L")
img = np.array(image)

# Edge Detection using Canny
edge = cv2.Canny(img, 50, 150)
"""

    st.image(edge, caption=f"Edge Detection using {method}", use_container_width=False)
    st.code(code_str, language="python")
