from turtle import onclick
import streamlit as st
import streamlit.components.v1 as components
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML
from typing import Callable, List

def convert_to_bw(img, thresh):
    # Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold (black / white)
    # Second param is adjustable
    ret, threshold = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

    return ret, threshold

def read_contours(img, threshold, contour_thresh):

    contours, hierarchy = cv2.findContours(image=threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    img_copy = img.copy()
    cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    xs = []
    ys = []

    # Get points for all outlines except for first because that's a weird square we don't want
    # CV2 docs weren't that helpful :(
    for contour in contours[contour_thresh:]:
        # We will reshape to 1D
        xs.extend(contour[:, :, 0].reshape(-1,))
        ys.extend(contour[:, :, 1].reshape(-1,))

    xs = np.array(xs)
    ys =- np.array(ys) # otherwise we will get a cursed, upside down photo

    return xs, ys, contours

def trapz(f: Callable[[float], float], l: float, r: float, n: int) -> float:
    """
    Numerical integration method from class
    """
    dx: float = (r - l) / n
    midsum: float = 2 * sum([f(l + i * dx) for i in range(1, n)])
    return 0.5 * dx * (f(l) + midsum + f(r))

def generate_coefficients(n: int, lg: Callable[[float], float]) -> List[float]:
    """
    Generates a list of coefficients for vectors
    """
    cs: List[float] = []
    for i in range(-n, n+1):
        g: Callable[[float], float] = lambda t : lg(t) * np.exp(-i * t * 1j)
        
        c_n: float = 1/(2*np.pi) * trapz(g, 0, 2*np.pi, 2000)
        cs.append(c_n)
    return cs

def sort_cs(cs: List[float], n: int) -> List[float]:
    sortedcs = [cs[n]]
    [sortedcs.extend([cs[n+i], cs[n-i]]) for i in range(1, n+1)]
        
    return np.array(sortedcs)



st.title("Fourier Drawing")
uploaded = st.file_uploader("Choose image", type=["png", "jpeg", "jpg"])

if uploaded is not None:
    _, cen, _  =st.columns(3)

    with cen:
        st.image(uploaded, width=300)
    #st.image(cv2.cvtColor(threshold, cv2.COLOR_BGR2RGB))
    st.header("Adjust image settings")
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, 1) #cv2.imread(uploaded)
    thresh = 0
    contour_thresh = 0

    _, threshold = convert_to_bw(img, thresh)
    xs, ys, contours = read_contours(img, threshold, contour_thresh)

    col1, col2 = st.columns(2)
    with col1:
        thresh = st.slider("Pixel threshold", 0, 255)
    with col2:
        contour_thresh = st.slider("Contour selection", 0, len(contours), value=1)

    col1, col2 = st.columns(2)

    _, threshold = convert_to_bw(img, thresh)
    xs, ys, contours = read_contours(img, threshold, contour_thresh)
    xs, ys = xs - xs.mean(), ys - ys.mean()
    
    outline = plt.figure()
    plt.plot(xs, ys, markersize=0.5)
    plt.axis('equal')
    xlim_data = plt.xlim() 
    ylim_data = plt.ylim()
    st.pyplot(outline)

    st.subheader("Coefficient Generation")
    order = st.slider("Number of vectors", 1, 100)
    generate_btn = st.button("Generate coefficients")

    if (generate_btn):
        time_arr = np.linspace(0, 2 * np.pi, len(xs))
        lg = lambda t : np.interp(t, time_arr, xs + 1j*ys)

        coefficients = generate_coefficients(order, lg)    

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 1), dpi=500);

        # Make sure we can see entire drawing and remove axis / make aspect ratio less thicc
        ax1.set_xlim(xlim_data[0]-50, xlim_data[1]+50);
        ax1.set_ylim(ylim_data[0]-50, ylim_data[1]+50);
        ax1.set_axis_off();
        ax1.set_aspect('equal');

        ax2.set_axis_off();
        ax2.set_aspect('equal');

        frames = 100

        # Create circles
        circles = [ax1.plot([], [], 'g-', linewidth=0.3)[0] for i in range(-order, order+1)];
        circles2 = [ax2.plot([], [], 'g-', linewidth=0.3)[0] for i in range(-order, order+1)];

        # Create vectors that are not arrows because matplotlib's arrow is v e r y  s l o w
        lines = [ax1.plot([], [], 'y-', linewidth=0.3)[0] for i in range(-order, order+1)];
        lines2 = [ax2.plot([], [], 'y-', linewidth=0.3)[0] for i in range(-order, order+1)];

        # Drawings for both axes
        drawing, = ax1.plot([], [], 'k-', linewidth=0.5);
        drawing2, = ax2.plot([], [], 'k-', linewidth=0.5);
        draw_x, draw_y = [], []

        def draw_circle(k, c_real, c_imag, center_x, center_y):
            """
            Draws a circle with radius equal to length of vector and center at starting point of vector
            """
            r = np.linalg.norm([c_real, c_imag])
            theta = np.linspace(0, 2*np.pi, 50);
            
            x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
            
            circles[k].set_data(x, y);
            circles2[k].set_data(x, y);
            
        def draw_line(k, c_real, c_imag, center_x, center_y):
            """
            Draws a rotating vector in the current frame
            """
            x, y = [center_x, center_x + c_real], [center_y, center_y + c_imag]
            
            lines[k].set_data(x, y);
            lines2[k].set_data(x, y);

        def animiter(i, time_arr, cs):
            """
            Animates individual frames
            """
            
            # Multiplying 'rotates' the vectors
            cs = sort_cs(cs * np.array([np.exp(j*time_arr[i]*1j) for j in range(-order, order+1)]), order)
            
            # Gets the vector coordinates
            cs_real = cs.real
            cs_imag = cs.imag
            
            # We shall begin at origin
            center_x, center_y = 0, 0
            
            # Draw vectors/circles based on coefficients * the exponential part at this point in the time array
            for k, (c_real, c_imag) in enumerate(zip(cs_real, cs_imag)):
                draw_circle(k, c_real, c_imag, center_x, center_y)
                draw_line(k, c_real, c_imag, center_x, center_y)
                
                # Update so next vector starts on the tip of the previous
                center_x, center_y = center_x + c_real, center_y + c_imag
                
            draw_x.extend([center_x])
            draw_y.extend([center_y])
            
            # Z o o m
            ax2.set_xlim(center_x-100, center_x+100);
            ax2.set_ylim(center_y-100, center_y+100);
            
            drawing.set_data(draw_x, draw_y);
            drawing2.set_data(draw_x, draw_y);

        matplotlib.rcParams['animation.embed_limit'] = 52_428_800.

        frames=300
        time_arr = np.linspace(0, 2*np.pi, num=frames)
        anim = animation.FuncAnimation(fig, animiter, frames=frames, fargs=(time_arr, coefficients), interval=20)
        with open("myvideo.html","w") as f:
            print(anim.to_html5_video(), file=f)
        
        HtmlFile = open("myvideo.html", "r")
        source_code = HtmlFile.read() 
        components.html(source_code, height = 1000,width=1000)
        generate_btn = False