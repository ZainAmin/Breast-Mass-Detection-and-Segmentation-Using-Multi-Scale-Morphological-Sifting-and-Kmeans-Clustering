import sys
sys.path.append('..\\helpers')

import cv2
import os
import numpy as np
import math

from morphological_sifter import MorphologicalSifter
from preprocessing import Preprocessor

from functools import reduce
from pywt import dwt2
from tqdm import tqdm, tqdm_notebook

# From helpers directory
import display

from sklearn.cluster import KMeans
import warnings

class Segmentation:
    def __init__(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.mms = MorphologicalSifter()
        self.preprocessor = Preprocessor()


    def slico(self, img):
        # Apply Superpixel SLIC algorithm
        algo = cv2.ximgproc.createSuperpixelSLIC(img, cv2.ximgproc.SLICO, 10)
        algo.iterate(10)

        # Get labels and number of superpixels
        labels = algo.getLabels()
        num_superpixels = algo.getNumberOfSuperpixels()

        # Calculate superpixel means
        superpixel_means = np.zeros((num_superpixels, 3))
        for k in range(num_superpixels):
            mask = (labels == k).astype('uint8')
            superpixel_means[k] = cv2.mean(img, mask)[:3]

        # Calculate superpixel areas
        # superpixel_areas = np.bincount(labels.flatten())

        return superpixel_means, num_superpixels, labels
    
    def mean_shift_filter(self, image, spatial_radius=15, range_radius=60):
        """
        Apply mean shift filter to a grayscale image.

        Parameters:
        - image: 2D numpy array of type uint16 with shape (height, width)
        - spatial_radius: int, spatial distance (in pixels) to consider for the filter
        - range_radius: int, color distance to consider for the filter

        Returns:
        - filtered_image: 2D numpy array of type uint16 with shape (height, width),
                        the result of the filter
        """

        # Convert image to float32 for better precision
        image_float = image.astype(np.float32)

        # Convert image to uint8 for OpenCV
        image_uint8 = cv2.convertScaleAbs(image_float / np.max(image_float) * 255)

        # Convert 2-channel grayscale image to 3-channel grayscale image
        image_3ch = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

        # Apply mean shift filter using OpenCV
        filtered_image_uint8 = cv2.pyrMeanShiftFiltering(image_3ch, spatial_radius, range_radius)

        # Convert the filtered image back to uint16
        filtered_image_float = cv2.cvtColor(filtered_image_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 * np.max(image_float)
        filtered_image = filtered_image_float.astype(np.uint16)

        return filtered_image

    def eliminate_thin_lines(self, image):
        # Convert the image to grayscale if necessary
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply binary thresholding to obtain a binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Define the kernel for morphological operations
        kernel = np.ones((15, 15), np.uint8)

        # Perform opening operation
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return opened

    def eliminate(self, image):
        
        # Normalize the filtered image to improve contrast
        normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Perform adaptive thresholding on the normalized image
        thresh = cv2.adaptiveThreshold(normalized_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    #     _, thresh = cv2.threshold(normalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Perform morphological operations
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
    #     # Perform morphological opening to eliminate remaining small objects
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Find contours of thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a blank image to draw contours on
        result = np.zeros_like(thresh)
        
        min_area = 110
        max_area = 551132

        # Loop through contours and eliminate those outside the desired area range
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area and area < max_area:
                cv2.drawContours(result, [cnt], 0, (255, 255, 255), -1)
                
        threshold = 128
        _, binary_result = cv2.threshold(self.eliminate_thin_lines(result), threshold, 255, cv2.THRESH_BINARY)


        return binary_result

    def kmeans(self, input_img):
        try:        
            # convert to float & reshape to a [1 x W*H] Mat
            # (so every pixel is on a row of its own)
            data = input_img.astype(np.float32)
            data = data.reshape(1, -1)

            # do kmeans
            # Tyry higher k > 2 - K=2 is binarization
            _, labels, centers = cv2.kmeans(
                data, 2, None,
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0),
                attempts=15,
                flags=cv2.KMEANS_PP_CENTERS
            )

            # replace pixel values with their center value:
            p = data[0]
            for i in range(data.shape[1]):
                center_id = labels[i]
                p[i] = centers[center_id]

            # back to 2D, and uchar:
            img = data.reshape(input_img.shape)
            img = np.clip(img, 0, 65535).astype(np.uint16)
            
            # Normalize the filtered image to improve contrast
            

            # Perform adaptive thresholding on the normalized image
    #         img = cv2.adaptiveThreshold(normalized_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2)        
    #         _, img = cv2.threshold(normalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return img

        except Exception as e:
            print(str(e))


    def pipeline(self, input_image):
        
        sifted_image = self.mms.multi_scale_morphological_sifters(input_image,10,18,15,3689,70/1000)

        # self.preprocessor.save_image(self.preprocessor.prepare_export(sifted_image[2]), r'../report/report_images/mms/', 'sifted_image_2.jpg')
        # self.preprocessor.save_image(self.preprocessor.prepare_export(sifted_image[3]), r'../report/report_images/mms/', 'sifted_image_3.jpg')
        # self.preprocessor.save_image(self.preprocessor.prepare_export(sifted_image[4]), r'../report/report_images/mms/', 'sifted_image_4.jpg')
        # self.preprocessor.save_image(self.preprocessor.prepare_export(sifted_image[5]), r'../report/report_images/mms/', 'sifted_image_5.jpg')
        # self.preprocessor.save_image(self.preprocessor.prepare_export(sifted_image[9]), r'../report/report_images/mms/', 'sifted_image_9.jpg')

        # Apply the mean shift filtering to the different scales of sifted images
        mean_shifted_image_0 = self.mean_shift_filter(sifted_image[0])
        mean_shifted_image_1 = self.mean_shift_filter(sifted_image[1])
        mean_shifted_image_2 = self.mean_shift_filter(sifted_image[2])
        mean_shifted_image_3 = self.mean_shift_filter(sifted_image[3])
        mean_shifted_image_4 = self.mean_shift_filter(sifted_image[4])
        mean_shifted_image_5 = self.mean_shift_filter(sifted_image[5])
        mean_shifted_image_6 = self.mean_shift_filter(sifted_image[6])
        mean_shifted_image_7 = self.mean_shift_filter(sifted_image[7])
        mean_shifted_image_8 = self.mean_shift_filter(sifted_image[8])
        mean_shifted_image_9 = self.mean_shift_filter(sifted_image[9])

        # self.preprocessor.save_image(mean_shifted_image_2, r'../report/report_images/mean_shift_filter/', 'mean_shifted_image_2.tif')
        # self.preprocessor.save_image(mean_shifted_image_3, r'../report/report_images/mean_shift_filter/', 'mean_shifted_image_3.tif')
        # self.preprocessor.save_image(mean_shifted_image_4, r'../report/report_images/mean_shift_filter/', 'mean_shifted_image_4.tif')
        # self.preprocessor.save_image(mean_shifted_image_5, r'../report/report_images/mean_shift_filter/', 'mean_shifted_image_5.tif')
        # self.preprocessor.save_image(mean_shifted_image_9, r'../report/report_images/mean_shift_filter/', 'mean_shifted_image_9.tif')


        # print("mean_shifted_image_0: ", mean_shifted_image_0.shape, mean_shifted_image_0.dtype, mean_shifted_image_0.max())
        
        kmean0 = self.kmeans(mean_shifted_image_0) 
        kmean1 = self.kmeans(mean_shifted_image_1) 
        kmean2 = self.kmeans(mean_shifted_image_2) 
        kmean3 = self.kmeans(mean_shifted_image_3)
        kmean4 = self.kmeans(mean_shifted_image_4) 
        kmean5 = self.kmeans(mean_shifted_image_5) 
        kmean6 = self.kmeans(mean_shifted_image_6) 
        kmean7 = self.kmeans(mean_shifted_image_7) 
        kmean8 = self.kmeans(mean_shifted_image_8) 
        kmean9 = self.kmeans(mean_shifted_image_9)

        # self.preprocessor.save_image(self.preprocessor.prepare_export(kmean2), r'../report/report_images/kmeans/', 'kmeans_2.tif')
        # self.preprocessor.save_image(self.preprocessor.prepare_export(kmean3), r'../report/report_images/kmeans/', 'kmeans_3.tif')
        # self.preprocessor.save_image(self.preprocessor.prepare_export(kmean4), r'../report/report_images/kmeans/', 'kmeans_4.tif')
        # self.preprocessor.save_image(self.preprocessor.prepare_export(kmean5), r'../report/report_images/kmeans/', 'kmeans_5.tif')
        # self.preprocessor.save_image(self.preprocessor.prepare_export(kmean9), r'../report/report_images/kmeans/', 'kmeans_9.tif')

        mass0 = self.eliminate(kmean0) 
        mass1 = self.eliminate(kmean1) 
        mass2 = self.eliminate(kmean2) 
        mass3 = self.eliminate(kmean3)
        mass4 = self.eliminate(kmean4) 
        mass5 = self.eliminate(kmean5) 
        mass6 = self.eliminate(kmean6) 
        mass7 = self.eliminate(kmean7)
        mass8 = self.eliminate(kmean8) 
        mass9 = self.eliminate(kmean9)

        # self.preprocessor.save_image(mass2, r'../report/report_images/elimination/', 'mass_2.tif')
        # self.preprocessor.save_image(mass3, r'../report/report_images/elimination/', 'mass_3.tif')
        # self.preprocessor.save_image(mass4, r'../report/report_images/elimination/', 'mass_4.tif')
        # self.preprocessor.save_image(mass5, r'../report/report_images/elimination/', 'mass_5.tif')
        # self.preprocessor.save_image(mass9, r'../report/report_images/elimination/', 'mass_9.tif')
        
        return sifted_image, [mean_shifted_image_0, mean_shifted_image_1, mean_shifted_image_2, mean_shifted_image_3, mean_shifted_image_4, mean_shifted_image_5, mean_shifted_image_6, mean_shifted_image_7, mean_shifted_image_8, mean_shifted_image_9], [kmean0, kmean1, kmean2, kmean3, kmean4, kmean5, kmean6, kmean7, kmean8, kmean9], \
                [mass0, mass1, mass2, mass3, mass4, mass5, mass6, mass7, mass8, mass9]


