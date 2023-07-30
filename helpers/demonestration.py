import os
import cv2
import numpy as np

def is_folder_not_empty(folder_path):
    return len(os.listdir(folder_path)) > 0

def are_folder_file_names_equal(folder_path1, folder_path2):
    file_names1 = sorted(os.listdir(folder_path1))
    file_names2 = sorted(os.listdir(folder_path2))
    return file_names1 == file_names2


def upsample(image):
    height, width = image.shape[:2]
    new_height = int(height * 4)
    new_width = int(width * 4)
    
    return cv2.resize(image.copy(), (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def binarize(image):
    threshold = 128
    _, binarized_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binarized_image

def show_image(labe, image):    
    cv2.imshow(labe, cv2.resize(image, (image.shape[1]//3, image.shape[0]//3)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_to_8bit(image_16bit):
    if not isinstance(image_16bit, np.ndarray) or image_16bit.dtype != np.uint16:
        raise ValueError("Input must be a 16-bit numpy array")

    # Scale the 16-bit image to the range of 0-255
    image_8bit = (image_16bit / np.max(image_16bit) * 255).astype(np.uint8)

    return image_8bit
