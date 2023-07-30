import cv2
import numpy as np
import math

class MorphologicalSifter:
    def __init__(self):
        pass
    
    def generate_rotated_kernel(self, D, delta_theta):
        """
        Generates a rotated kernel for morphological operations.

        Args:
            D (int): Outer/Inner diameter of the kernel.
            delta_theta (float): Rotation angle in degrees.

        Returns:
            np.ndarray: Rotated kernel.
        """
        kernel = np.zeros((D, D), dtype=int)
        kernel[D // 2, :] = 1

        rotmat = cv2.getRotationMatrix2D((int(D // 2), int(D // 2)), delta_theta, 1.0)
        rotated_kernel = cv2.warpAffine(np.uint8(kernel), rotmat, (D, D))

        rotated_kernel = rotated_kernel[:, np.any(rotated_kernel, axis=0)]
        rotated_kernel = rotated_kernel[np.any(rotated_kernel, axis=1)]

        return rotated_kernel

    # Multi-Scale Morphological Sifting
    def multi_scale_morphological_sifters(self, input_img, n_scale, n_lse_elements, area_min, area_max, pixel_size):
        """
        Performs multi-scale morphological sifting on the input image.

        Args:
            input_img (np.ndarray): Input image.
            n_scale (int): Number of scales.
            n_lse_elements (int): Number of elements in the Line Structuring Element (LSE).
            area_min (float): Minimum area.
            area_max (float): Maximum area.
            pixel_size (float): Pixel size in the image.

        Returns:
            np.ndarray: Summed images after morphological sifting at each scale.
        """

        # Initialize variables
        SI = np.zeros((n_scale+1), dtype=float) # Scale Interval
        D1 = np.zeros((n_scale), dtype=int)     # Outer Diameter
        D2 = np.zeros((n_scale), dtype=int)     # Inner Diameter

        theta_range = range(0,180,int(180/n_lse_elements))

        SI[0] = 1.   # Minimum Dimention in First Iteration

        # Calculate Diameters D1, D2
        DImin = 2 * math.sqrt(area_min / math.pi) / (pixel_size * 4)
        DImax = 2 * math.sqrt(area_max / math.pi) / (pixel_size * 4)

        SI = np.logspace(math.log10(DImin), math.log10(DImax), n_scale+1)
        D1 = np.round(SI[:-1]).astype(int)
        D2 = np.round(SI[1:]).astype(int)

        # Ensure Odd-Numbered Diameter to use as Kernel Filter
        D1[D1 % 2 == 0] += 1
        D2[D2 % 2 == 0] += 1

        # print(D1) # [ 17  21  27  37  47  63  81 107 141 187]
        # print(D2) # [ 21  27  37  47  63  81 107 141 187 245]

        # Placeholder for Summing Image    
        summed_image = np.zeros((n_scale,input_img.shape[0],input_img.shape[1]), dtype=int)
        for i in range(n_scale):
            for delta_theta in theta_range:
                
                # Apply Top-Hat Transform using Outer Diameter Line
                rotated_kernel = self.generate_rotated_kernel(D2[i], delta_theta)
                dst1           = cv2.morphologyEx(input_img, cv2.MORPH_TOPHAT, rotated_kernel)
        
                # Apply Morphological Opening using Inner Diameter Line
                rotated_kernel = self.generate_rotated_kernel(D1[i], delta_theta)
                dst2           = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, rotated_kernel)
                
                summed_image[i,:,:] = summed_image[i,:,:] + dst2
            summed_image[i,:,:] = (summed_image[i,:,:]/np.max(summed_image[i,:,:]))*(2**16-1) # Normalize Output Image

        return summed_image

