import streamlit as st
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

def convert_to_bw(img, thresh):
    # Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold (black / white)
    # Second param is adjustable
    ret, threshold = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

    return ret, threshold

def read_contours(img, threshold):

    contours, hierarchy = cv2.findContours(image=threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    img_copy = img.copy()
    cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    #plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))

    xs = []
    ys = []

    # Get points for all outlines except for first because that's a weird square we don't want
    # CV2 docs weren't that helpful :(
    for contour in contours:
        # We will reshape to 1D
        xs.extend(contour[:, :, 0].reshape(-1,))
        ys.extend(contour[:, :, 1].reshape(-1,))

    xs = np.array(xs)
    ys =- np.array(ys) # otherwise we will get a cursed, upside down photo

    return xs, ys

st.title("Fourier Drawing")
uploaded = st.file_uploader("Choose image", type=["png", "jpeg", "jpg"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, 1) #cv2.imread(uploaded)
    thresh = st.slider("Pixel threshold", 0, 255)
    _, threshold = convert_to_bw(img, thresh)

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB))
    
    xs, ys = read_contours(img, threshold)
    outline = plt.figure()
    plt.plot(xs, ys, 'o', markersize=0.5)
    plt.axis('equal')
    with col2:
        st.pyplot(outline)

