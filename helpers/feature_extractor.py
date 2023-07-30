import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments, moments_hu
from skimage import feature

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
        distances = [1]
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
            # hu_moments = cv2.HuMoments(m)
            # hu_moments = np.ravel(hu_moments)
            
            # Add Hu Moments to feature vector
            # shape_features.append(hu_moments)

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
            # bounding_box_area = w * h
            # extent = float(area) / bounding_box_area
            
            # Solidity (contour area / convex hull area)
            # convex_hull = cv2.convexHull(contours[0])
            # convex_hull_area = cv2.contourArea(convex_hull)
            # solidity = float(area) / convex_hull_area
            
            # Compactness (perimeter^2 / contour area)
            compactness = (perimeter ** 2) / area
            
            # Add all features to the feature vector
            shape_features.append([area, perimeter, aspect_ratio, compactness])
        
        # Concatenate features into a single feature vector
        shape_features = np.concatenate(shape_features)
        
        return shape_features

    def extract_intensity_features(self, image, labels):
        # average intenssity and standard dev in the mass here <- easy features

        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Input must be a 8-bit numpy array")
        
        # Initialize feature vector
        intensity_features = []

        for label in np.unique(labels):
            # Extract pixel values
            pixels = image[labels == label]
                    
            # Calculate mean and standard deviation of pixel intensities
            mean_intensity = np.mean(pixels)
            std_intensity = np.std(pixels)
            
            # Add mean and standard deviation to feature vector
            intensity_features.append([mean_intensity, std_intensity])
            
        intensity_features = np.concatenate(intensity_features)
        
        return intensity_features
    
    def uniform_lbp(self, image, radius, num_points):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Input must be an 8-bit numpy array")

        # Convert image to grayscale
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = image

        # Apply LBP operation
        lbp = np.zeros_like(gray_image, dtype=np.uint8)
        for i in range(num_points):
            theta = 2 * np.pi * i / num_points
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            sample_x = np.round(x).astype(int)
            sample_y = np.round(y).astype(int)
            sample = cv2.getRectSubPix(gray_image, (radius * 2 + 1, radius * 2 + 1), (x + radius, y + radius))
            lbp = np.bitwise_or(lbp, (sample > gray_image + 1e-6) << i)

        # Compute uniform LBP pattern
        uniform_lbp = np.zeros_like(lbp)
        for i in range(num_points):
            pattern = ((lbp >> i) & 1) | ((lbp >> ((i + 1) % num_points)) & 1) << 1 | ((lbp >> ((i + 2) % num_points)) & 1) << 2 | ((lbp >> ((i + 3) % num_points)) & 1) << 3
            uniform_pattern = np.where(pattern < 3, pattern, 0)
            uniform_lbp = np.bitwise_or(uniform_lbp, uniform_pattern << i)

        # Calculate histogram of uniform LBP values
        hist, _ = np.histogram(uniform_lbp.ravel(), bins=np.arange(0, num_points + 2), range=(0, num_points + 1))

        # Normalize histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return hist
    
    def lbp_features(self, image, P=[8, 16], R=[1, 2]):
        features = []
        
        for p in P:
            for r in R:
                lbp = feature.local_binary_pattern(image, p, r, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, p + 3), range=(0, p + 2))
                features.extend(hist)
        
        return np.array(features)
    
    