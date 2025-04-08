import numpy as np
import cv2
import streamlit as st 
from PIL import Image 



def apply_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft) 
    return dft_shift
def apply_ifft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)  
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    return img_back

def apply_ifft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)  
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    return img_back

def create_filter(shape, cutoff, filter_type="low", order=2):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center
    mask = np.zeros((rows, cols), np.float32)

    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow)**2 + (v - ccol)**2)
            if filter_type == "low":
                if d <= cutoff:
                    mask[u, v] = 1
                else:
                    mask[u, v] = 0
            elif filter_type == "high":
                if d >= cutoff:
                    mask[u, v] = 1
                else:
                    mask[u, v] = 0

    return mask

with subtab[1]:
    subsubtab = st.tabs(["Select",'Ideal LPF','Gaussian LPF','ButterWorth LPF'])
    with subsubtab[1]:
        st.write("Apply Ideal Low Pass Filter")
        img=Image.open(uploaded_file).convert("L")
        img_array=np.array(img)
        fft_image = apply_fft(img_array)
        ilpf = create_filter(img_array.shape, 30, "low")
        filtered_fft = fft_image * ilpf
        ifft_image = apply_ifft(filtered_fft)

            # Normalize image for display
        ifft_image = (ifft_image - ifft_image.min()) / (ifft_image.max() - ifft_image.min()) * 255
        ifft_image = np.uint8(ifft_image)

        st.image(ifft_image,caption="Ideal Low Pass Filtered Image",use_container_width=False)
        st.code("""
import numpy as np
from PIL import Image
def apply_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft) 
    return dft_shift
def apply_ifft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)  
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    return img_back
def create_filter(shape, cutoff, filter_type="low", order=2):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center
    mask = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow)**2 + (v - ccol)**2)
            if filter_type == "low":
                if d <= cutoff:
                    mask[u, v] = 1
                else:
                    mask[u, v] = 0
    return mask
img=Image.open("path")
img_array=np.array(img)
fft_image = apply_fft(img_array)
ilpf = create_filter(img_array.shape, 30, "low")
filtered_fft = fft_image * ilpf
ifft_image = apply_ifft(filtered_fft)""",language="python")


    with subsubtab[2]:
        st.write("Apply Gaussian Low Pass Filter")
        img=Image.open(uploaded_file).convert("L")
        img_array=np.array(img)
        fft_image = apply_fft(img_array)
        ilpf = np.exp(-((np.arange(img_array.shape[0])[:, None] - img_array.shape[0] // 2) ** 2 +
                (np.arange(img_array.shape[1]) - img_array.shape[1] // 2) ** 2) / (2 * 30 ** 2)) 
        filtered_fft = fft_image * ilpf
        ifft_image = apply_ifft(filtered_fft)

            # Normalize image for display
        ifft_image = (ifft_image - ifft_image.min()) / (ifft_image.max() - ifft_image.min()) * 255
        ifft_image = np.uint8(ifft_image)

        st.image(ifft_image,caption="Gaussian Low Pass Filtered Image",use_container_width=False)
        st.code("""
import numpy as np
from PIL import Image
def apply_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft) 
    return dft_shift
def apply_ifft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)  
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    return img_back
img=Image.open("path")
img_array=np.array(img)
fft_image = apply_fft(img_array)
ilpf = np.exp(-((np.arange(img_array.shape[0])[:, None] - img_array.shape[0] // 2) ** 2 +
                (np.arange(img_array.shape[1]) - img_array.shape[1] // 2) ** 2) / (2 * 30 ** 2)) 
filtered_fft = fft_image * ilpf
ifft_image = apply_ifft(filtered_fft)
""",language="python")
        

    with subsubtab[3]:
        st.write("Apply Butterworth Low Pass Filter")
        img=Image.open(uploaded_file).convert("L")
        img_array=np.array(img)
        fft_image = apply_fft(img_array)
        ilpf =  1 / (1 + (np.sqrt((np.arange(img_array.shape[0])[:, None] - img_array.shape[0] // 2) ** 2 +
                         (np.arange(img_array.shape[1]) - img_array.shape[1] // 2) ** 2) / 30) ** (2 * 2))
        filtered_fft = fft_image * ilpf
        ifft_image = apply_ifft(filtered_fft)

            # Normalize image for display
        ifft_image = (ifft_image - ifft_image.min()) / (ifft_image.max() - ifft_image.min()) * 255
        ifft_image = np.uint8(ifft_image)

        st.image(ifft_image,caption="Butterworth Low Pass Filtered Image",use_container_width=False)
        st.code("""import numpy as np
from PIL import Image
def apply_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft) 
    return dft_shift
def apply_ifft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)  
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    return img_back
img=Image.open("path")
img_array=np.array(img)
fft_image = apply_fft(img_array)
ilpf =  1 / (1 + (np.sqrt((np.arange(img_array.shape[0])[:, None] - img_array.shape[0] // 2) ** 2 +
                         (np.arange(img_array.shape[1]) - img_array.shape[1] // 2) ** 2) / 30) ** (2 * 2))
filtered_fft = fft_image * ilpf
ifft_image = apply_ifft(filtered_fft)""",language="python")
        





with subtab[2]:
    subsubtab = st.tabs(["Select",'Ideal HPF','Gaussian HPF','ButterWorth HPF'])
    with subsubtab[1]:
        st.write("Apply Ideal High Pass Filter")
        img=Image.open(uploaded_file).convert("L")
        img_array=np.array(img)
        fft_image = apply_fft(img_array)
        ilpf = create_filter(img_array.shape, 30, "high")
        filtered_fft = fft_image * ilpf
        ifft_image = apply_ifft(filtered_fft)

            # Normalize image for display
        ifft_image = (ifft_image - ifft_image.min()) / (ifft_image.max() - ifft_image.min()) * 255
        ifft_image = np.uint8(ifft_image)

        st.image(ifft_image,caption="Ideal Low Pass Filtered Image",use_container_width=False)
        st.code("""
import numpy as np
from PIL import Image
def apply_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft) 
    return dft_shift
def apply_ifft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)  
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    return img_back
def create_filter(shape, cutoff, filter_type="low", order=2):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2  # Center
    mask = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow)**2 + (v - ccol)**2)
            if filter_type == "high":
                if d <= cutoff:
                    mask[u, v] = 0
                else:
                    mask[u, v] = 1
    return mask
img=Image.open("path")
img_array=np.array(img)
fft_image = apply_fft(img_array)
ilpf = create_filter(img_array.shape, 30, "high")
filtered_fft = fft_image * ilpf
ifft_image = apply_ifft(filtered_fft)""",language="python")




    with subsubtab[2]:
        st.write("Apply Gaussian High Pass Filter")
        img=Image.open(uploaded_file).convert("L")
        img_array=np.array(img)
        fft_image = apply_fft(img_array)
        ilpf = np.exp(-((np.arange(img_array.shape[0])[:, None] - img_array.shape[0] // 2) ** 2 +
                (np.arange(img_array.shape[1]) - img_array.shape[1] // 2) ** 2) / (2 * 30 ** 2)) 
        ilpf=1-ilpf
        filtered_fft = fft_image * ilpf
        ifft_image = apply_ifft(filtered_fft)

            # Normalize image for display
        ifft_image = (ifft_image - ifft_image.min()) / (ifft_image.max() - ifft_image.min()) * 255
        ifft_image = np.uint8(ifft_image)

        st.image(ifft_image,caption="Gaussian High Pass Filtered Image",use_container_width=False)
        st.code("""
import numpy as np
from PIL import Image
def apply_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft) 
    return dft_shift
def apply_ifft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)  
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    return img_back
img=Image.open("path")
img_array=np.array(img)
fft_image = apply_fft(img_array)
ilpf = np.exp(-((np.arange(img_array.shape[0])[:, None] - img_array.shape[0] // 2) ** 2 +
                (np.arange(img_array.shape[1]) - img_array.shape[1] // 2) ** 2) / (2 * 30 ** 2)) 
ilpf=1-lpf
filtered_fft = fft_image * ilpf
ifft_image = apply_ifft(filtered_fft)
""",language="python")
        

    with subsubtab[3]:
        st.write("Apply Butterworth High Pass Filter")
        img=Image.open(uploaded_file).convert("L")
        img_array=np.array(img)
        fft_image = apply_fft(img_array)
        ilpf =  1 / (1 + (np.sqrt((np.arange(img_array.shape[0])[:, None] - img_array.shape[0] // 2) ** 2 +
                         (np.arange(img_array.shape[1]) - img_array.shape[1] // 2) ** 2) / 30) ** (2 * 2))
        ilpf=1-ilpf
        filtered_fft = fft_image * ilpf
        ifft_image = apply_ifft(filtered_fft)

            # Normalize image for display
        ifft_image = (ifft_image - ifft_image.min()) / (ifft_image.max() - ifft_image.min()) * 255
        ifft_image = np.uint8(ifft_image)

        st.image(ifft_image,caption="Butterworth High Pass Filtered Image",use_container_width=False)
        st.code("""import numpy as np
from PIL import Image
def apply_fft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft) 
    return dft_shift
def apply_ifft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)  
    img_back = np.fft.ifft2(dft_ishift)
    img_back = np.abs(img_back)
    return img_back
img=Image.open("path")
img_array=np.array(img)
fft_image = apply_fft(img_array)
ilpf =  1 / (1 + (np.sqrt((np.arange(img_array.shape[0])[:, None] - img_array.shape[0] // 2) ** 2 +
                         (np.arange(img_array.shape[1]) - img_array.shape[1] // 2) ** 2) / 30) ** (2 * 2))
ilpf=1-ilpf
filtered_fft = fft_image * ilpf
ifft_image = apply_ifft(filtered_fft)""",language="python")
        
