import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments, moments_hu

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_texture_features(self, image):
        
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Input must be a 8-bit numpy array")
        
        # Convert image to grayscale
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = image
        
        # Compute GLCM matrix
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_image, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Extract texture features from GLCM matrix
        features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        texture_features = np.concatenate([graycoprops(glcm, feature).ravel() for feature in features])
        
        return texture_features

    def extract_shape_features(self, image, labels):
         
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Input must be a 8-bit numpy array")
        
        # Initialize feature vector
        shape_features = []
        
        # Loop over each superpixel
        for label in np.unique(labels):
            # Extract binary mask of superpixel
            mask = np.zeros(image.shape[:2], dtype="uint8")
            mask[labels == label] = 255
            
            # Calculate moments of superpixel
            m = cv2.moments(mask)
            
            # Calculate Hu Moments of superpixel
            hu_moments = cv2.HuMoments(m)
            hu_moments = np.ravel(hu_moments)
            
            # Add Hu Moments to feature vector
            shape_features.append(hu_moments)

            # Calculate additional shape features
            # Area
            area = m["m00"]
            
            # Perimeter
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True)
            
            # Bounding box dimensions
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Aspect ratio (width/height)
            aspect_ratio = float(w) / h
            
            # Extent (contour area / bounding box area)
            bounding_box_area = w * h
            extent = float(area) / bounding_box_area
            
            # Solidity (contour area / convex hull area)
            convex_hull = cv2.convexHull(contours[0])
            convex_hull_area = cv2.contourArea(convex_hull)
            solidity = float(area) / convex_hull_area
            
            # Compactness (perimeter^2 / contour area)
            compactness = (perimeter ** 2) / area
            
            # Add all features to the feature vector
            shape_features.append(np.concatenate([hu_moments, [area, perimeter, aspect_ratio, extent, solidity, compactness]]))
        
        # Concatenate features into a single feature vector
        shape_features = np.concatenate(shape_features)
        
        return shape_features

    def extract_intensity_features(self, image, labels):
         
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Input must be a 8-bit numpy array")
        
        # Initialize feature vector
        intensity_features = []
        
        # Loop over each superpixel
        for label in np.unique(labels):
            # Extract pixel values of superpixel
            pixels = image[labels == label]
                    
            # Calculate mean and standard deviation of pixel intensities
            mean_intensity = np.mean(pixels)
            std_intensity = np.std(pixels)
            
            # Add mean and standard deviation to feature vector
            intensity_features.append([mean_intensity, std_intensity])
            
        intensity_features = np.concatenate(intensity_features)
        
        return intensity_features
    
