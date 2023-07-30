import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(18, 6))
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap='gray')
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

def read_img(folder, img_name):

    return cv2.imread(
        os.path.join(
            os.path.dirname(f"..\\dataset\\{folder}\\"),
            img_name
        ),
        cv2.IMREAD_GRAYSCALE
    )

def convert_16_to_8_bit(image):
    # Ensure that the image is a 16-bit numpy array
    if not isinstance(image, np.ndarray) or image.dtype != np.uint16:
        raise ValueError("Input must be a 16-bit numpy array")

    # Normalize the pixel values to [0, 1] range
    image = image.astype(np.float32)
    image = image / 65535.0

    # Scale the pixel values to [0, 255] range
    image = np.round(image * 255).astype(np.uint8)

    return image


def convert_to_16bit(img):
    # Check if the image is already 16-bit
    if img.dtype == "uint16":
        return img
    else:
        # Convert the image to 16-bit
        img_16bit = (img.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
        return img_16bit

