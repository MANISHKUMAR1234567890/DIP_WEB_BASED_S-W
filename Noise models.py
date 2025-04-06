import numpy as np
import cv2 
import streamlit as st  
from PIL import Image


#
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image


# Function to add Rayleigh Noise
def add_rayleigh_noise(image, sigma=30):
    row, col, ch = image.shape
    rayleigh = np.random.rayleigh(sigma, (row, col, ch))
    noisy_image = np.clip(image + rayleigh, 0, 255).astype(np.uint8)
    return noisy_image

# Function to add Erlang (Gamma) Noise
def add_erlang_noise(image, k=2, theta=20):
    row, col, ch = image.shape
    erlang = np.random.gamma(k, theta, (row, col, ch))
    noisy_image = np.clip(image + erlang, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = np.copy(image)
    row, col, ch = image.shape

    # Salt noise (white pixels)
    num_salt = np.ceil(salt_prob * row * col)
    salt_coords = [np.random.randint(0, i, int(num_salt)) for i in (row, col)]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255

    # Pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * row * col)
    pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in (row, col)]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0

    return noisy_image

with subtab[1]:
    st.write("Add gaussian noise: ")
    image = Image.open(uploaded_file).convert("RGB") 
    image_np = np.array(image)
    g_n=add_gaussian_noise(image_np,mean=0, sigma=25)
    st.image(g_n,caption="Gaussian Noisy image",use_column_width=False)
    st.code(""" 
            import numpy as np
            from PIL import Image
            img=Image.open("path").convert("RGB")
            img_array=np.array(img)
            #add gaussian noise
            def add_gaussian_noise(image, mean=0, sigma=25):
                row, col, ch = image.shape
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
                return noisy_image
            gn=add_gaussian_noise(img_array,mean=0,sigma=25)""",language="python")



with subtab[2]:
    st.write("Add rayleigh noise: ")
    image = Image.open(uploaded_file).convert("RGB") 
    image_np = np.array(image)
    g_n=add_rayleigh_noise(image_np, sigma=30)
    st.image(g_n,caption="Rayleigh Noisy image",use_column_width=False)
    st.code(""" 
            import numpy as np
            from PIL import Image
            img=Image.open("path").convert("RGB")
            img_array=np.array(img)
            #add rayleigh noise
            def add_rayleigh_noise(image, sigma=30):
                row, col, ch = image.shape
                rayleigh = np.random.rayleigh(sigma, (row, col, ch))
                noisy_image = np.clip(image + rayleigh, 0, 255).astype(np.uint8)
                return noisy_image
            g_n=add_rayleigh_noise(img_array, sigma=30)""",language="python")


with subtab[3]:
    st.write("Add Erlang noise: ")
    image = Image.open(uploaded_file).convert("RGB") 
    image_np = np.array(image)
    g_n=add_erlang_noise(image_np, k=2, theta=20)
    st.image(g_n,caption="Erlang Noisy image",use_column_width=False)
    st.code(""" 
            import numpy as np
            from PIL import Image
            img=Image.open("path").convert("RGB")
            img_array=np.array(img)
            #add erlang noise
            def add_erlang_noise(image, k=2, theta=20):
                row, col, ch = image.shape
                erlang = np.random.gamma(k, theta, (row, col, ch))
                noisy_image = np.clip(image + erlang, 0, 255).astype(np.uint8)
                return noisy_image
            g_n=add_erlang_noise(img_array, k=2, theta=20)""",language="python")


with subtab[4]:
    st.write("Add salt and pepper noise: ")
    image = Image.open(uploaded_file).convert("RGB") 
    image_np = np.array(image)
    g_n=add_salt_and_pepper_noise(image_np, salt_prob=0.02, pepper_prob=0.02)
    st.image(g_n,caption="Salt and pepper Noisy image",use_column_width=False)
    st.code(""" 
            import numpy as np
            from PIL import Image
            img=Image.open("path").convert("RGB")
            img_array=np.array(img)
            #add Salt and pepper noise
            def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
                noisy_image = np.copy(image)
                row, col, ch = image.shape

                # Salt noise (white pixels)
                num_salt = np.ceil(salt_prob * row * col)
                salt_coords = [np.random.randint(0, i, int(num_salt)) for i in (row, col)]
                noisy_image[salt_coords[0], salt_coords[1], :] = 255

                # Pepper noise (black pixels)
                num_pepper = np.ceil(pepper_prob * row * col)
                pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in (row, col)]
                noisy_image[pepper_coords[0], pepper_coords[1], :] = 0

                return noisy_image
            g_n=add_salt_and_pepper_noise(img_array, salt_prob=0.02, pepper_prob=0.02)""",language="python")
