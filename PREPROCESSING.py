import cv2
import numpy as np
from PIL import Image
import streamlit as st

def convert_image(image, mode):
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if mode == "Grayscale":
        return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    elif mode == "Binary":
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return binary
    else:  # RGB
        return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

def normalization(image):
    img = np.array(image, dtype=np.float32)  # Convert to float for normalization
    normalized = cv2.normalize(img, None, alpha=200, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)




with subtab[1]:
            
    with st.form("conversion_form"):  # Use a form to refresh selection
        st.write("Perform Image Conversion")
        option = st.radio("Choose an Image Conversion Mode:", ("RGB", "Grayscale", "Binary"))
        apply_conversion = st.form_submit_button("Apply Conversion")
        a = st.checkbox("Generate Conversion Code")
        if apply_conversion:
            converted_img = convert_image(image, option)
            st.image(converted_img, caption=f"Converted Image: {option}", use_column_width=True)
            
            
        if a:
            if option == "Grayscale":
                        st.code("""
import cv2
import numpy as np

def convert_to_grayscale(image):
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    return gray
convert_to_grayscale("path")
        """, language="python")

            elif option == "Binary":
                    st.code("""
import cv2
import numpy as np

def convert_to_binary(image):
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary
convert_to_binary("path")
        """, language="python")

            elif option == "RGB":
                    st.code("""
import cv2
import numpy as np

def convert_to_rgb(image):
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return rgb
convert_to_rgb("path")
        """, language="python")

    with subtab[2]:
        st.write("Perform Image Normalization")

            
            
        n_converted_img =normalization(Image.open(uploaded_file))
        st.image(n_converted_img, caption=f"Normalized Image", use_column_width=True)
        b=st.button("Generate normalization code")
        if b:

            st.code("""
        import numpy as np
        import cv2
        def normalization(image):
            img = np.array(image, dtype=np.float32)  # Convert to float for normalization
            normalized = cv2.normalize(img, None, alpha=200, beta=255, norm_type=cv2.NORM_MINMAX)
            return normalized.astype(np.uint8)

        image=cv2.imread("your_image")
        normalize=normalization(image)""",language="python")
        with subtab[3]:
            st.write("Perform Image Cropping")
            crop_image=Image.open(uploaded_file)
            crop_image_array=np.array(crop_image)
            height, width = crop_image_array.shape[:2]

            x_start = st.slider("Select X Start", 0, width, 50)
            x_end = st.slider("Select X End", 0, width, 400)
            y_start = st.slider("Select Y Start", 0, height, 50)
            y_end = st.slider("Select Y End", 0, height, 300)
        
                

    # Ensure valid crop range
            if x_start < x_end and y_start < y_end:
        # Crop the image
                cropped_image = crop_image_array[y_start:y_end, x_start:x_end]
                
                st.image(cropped_image,caption=f"Cropped Image",use_column_width=True)
                c=st.button("Generate cropping code")
                if c:

                    st.code(""" 
                        import numpy as np
                        from PIL import Image
                        img=Image.open("path")
                        img_array=np.array(img)
                        height, width = img_array.shape[:2]
                        x_start="set your x-start value"
                        x_end="set your x-end value"
                        y_start="set your y-start value"
                        x_end="set your y-end value"
                        if x_start < x_end and y_start < y_end:

                            cropped_image = crop_image_array[y_start:y_end, x_start:x_end]
                        else:
                            print("Please ensure the start values are smaller than the end values")""",language='python')

            else:
                st.warning("Please ensure the start values are smaller than the end values.")


        with subtab[4]:
            st.write("Perform Image Resizing")
            image = np.array(Image.open(uploaded_file))
            new_width = st.slider("Select Width", 50, image.shape[1], image.shape[1]//2)
            new_height = st.slider("Select Height", 50, image.shape[0], image.shape[0]//2)
            resized_image = cv2.resize(image, (new_width, new_height))

    # Display the resized image
            st.image(resized_image, caption=f"Resized Image ({new_width}x{new_height})", use_column_width=False)
            d=st.button("Generate resizing code")
            if d:

                st.code("""
                    import cv2
                    import numpy as np
                    from PIL import Image
                    img=Image.open("path")
                    img_array=np.array(img)
                    new_width="enter width"
                    new_height="enter height"
                    resized_image = cv2.resize(image, (new_width, new_height))

                        """,language="python")


