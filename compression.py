import numpy as np
import cv2 
import streamlit as st  
from PIL import Image
import io

with subtab[1]:
    st.write("Apply lossless compression: ")
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Apply lossless PNG compression and store in memory
    encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # 9 = max lossless compression
    success, compressed_img = cv2.imencode('.png', image_bgr, encode_param)

    if success:
        # Convert compressed image bytes to a format Streamlit can display
        compressed_bytes = io.BytesIO(compressed_img.tobytes())
        compressed_pil = Image.open(compressed_bytes)

        # Display the compressed image in Streamlit
        st.image(compressed_pil, caption="Lossless Compressed Image",use_column_width=False)
        st.code("""
                import cv2
                import numpy as np
                from PIL import Image
                img=Image.open("path")
                img_array=np.array(img)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Apply lossless PNG compression and store in memory
                encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # 9 = max lossless compression
                success, compressed_img = cv2.imencode('.png', image_bgr, encode_param)

                if success:
                    # Convert compressed image bytes to a format Streamlit can display
                    compressed_bytes = io.BytesIO(compressed_img.tobytes())
                    compressed_pil = Image.open(compressed_bytes)
                else:
                    print("Compression failed)""",language='python')
    else:
        st.error("Compression failed.")
   



with subtab[2]:
    st.write("Apply lossy compression: ")
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Apply lossless PNG compression and store in memory
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 30] 
    success, compressed_img = cv2.imencode('.jpg', image_bgr, encode_param)

    if success:
        # Convert compressed image bytes to a format Streamlit can display
        compressed_bytes = io.BytesIO(compressed_img.tobytes())
        compressed_pil = Image.open(compressed_bytes)

        # Display the compressed image in Streamlit
        st.image(compressed_pil, caption="Lossy Compressed Image",use_column_width=False)
        st.code("""
                import cv2
                import numpy as np
                from PIL import Image
                img=Image.open("path")
                img_array=np.array(img)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Apply lossless PNG compression and store in memory
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 30]  
                success, compressed_img = cv2.imencode('.jpg', image_bgr, encode_param)

                if success:
                    # Convert compressed image bytes to a format Streamlit can display
                    compressed_bytes = io.BytesIO(compressed_img.tobytes())
                    compressed_pil = Image.open(compressed_bytes)
                else:
                    print("Compression failed)""",language='python')
    else:
        st.error("Compression failed.")
   



