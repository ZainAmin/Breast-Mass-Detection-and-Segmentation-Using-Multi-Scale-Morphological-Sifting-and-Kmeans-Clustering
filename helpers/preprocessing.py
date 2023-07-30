import cv2
from loguru import logger
import display
import glob
import os
import time
from tqdm import tqdm, tqdm_notebook
import numpy as np

from display import convert_to_16bit

class Preprocessor:
    def __init__(self):
        self._resized_img           = None
        self._gray_img              = None
        self._thresholding_mask     = None
        self._contour_img           = None
        self._segmented_img         = None
        self._gt_img                = None
        self._rescaled_img          = None
        self._clahe_img             = None
        self.scale_factor           = 4

    def crop_raw(self, image):
        raw_threshold   = self._threshold_mask(image)

        contours, _ = cv2.findContours(raw_threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped_image = image.copy()[y:y+h, x:x+w]

        # We have to resize as we resized in the pre-processed image. The scale factor removes floating point after the 
        # rescaling from the dimensions making the mask and segmentation sizes different.
        # Once rescaled, it will match the sizes 
        cropped_image = cv2.resize(cropped_image, None, fx=1/self.scale_factor, fy=1/self.scale_factor, interpolation=cv2.INTER_CUBIC)
        
        return cropped_image

    def _resize(self, images, gt_img):
        # logger.info(f"Resizing with a scale factor {self.scale_factor} INTER_CUBIC interpolation.")
        img = cv2.resize(images, None, fx=1/self.scale_factor, fy=1/self.scale_factor, interpolation=cv2.INTER_CUBIC)
        
        if(gt_img is not None):
            gt  = cv2.resize(gt_img, None, fx=1/self.scale_factor, fy=1/self.scale_factor, interpolation=cv2.INTER_CUBIC)
        else:
            gt = None
        return img, gt
    
    def _to_grayscale(self, images):
        return cv2.cvtColor(images.copy(), cv2.COLOR_BGR2GRAY)        
    
    def _threshold_mask(self, images):
        blur = cv2.GaussianBlur(images.copy(), (5,5), 0)
        return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    def _find_contours_and_segment(self, images, gt_img):
        
        contours, _ = cv2.findContours(self._thresholding_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        

        contour_image = cv2.drawContours(images.copy(), [largest_contour], -1, (255, 255, 255), 3)

        cropped_image = images.copy()[y:y+h, x:x+w]
        
        if(gt_img is not None):
            cropped_gt = gt_img.copy()[y:y+h, x:x+w]
        else:
            cropped_gt = None

        return contour_image, cropped_image, cropped_gt
    
    def _rescale(self, images):       
        return cv2.normalize(images.copy(), None, 0, 255, cv2.NORM_MINMAX)
    
    def _clahe(self, images):
        # clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(4,4))
        clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(4,4))
        return clahe.apply(images.copy())
    
    def _export_processed(self, dir, gt_dir, img_processed, gt_img, img_path):
        folder_directory = os.path.join(dir.split('\\')[0], dir.split('\\')[1], 'processed', dir.split('\\')[2])
        file_directory = os.path.join(folder_directory, img_path.split('\\')[-1])

        if not os.path.isdir(folder_directory):
            os.makedirs(folder_directory)
            logger.info(f"New directory created '{folder_directory}'")

        cv2.imwrite(file_directory ,img_processed)

        
        if(gt_img is not None):
            folder_directory_gt = os.path.join(dir.split('\\')[0], gt_dir.split('\\')[1], 'processed', gt_dir.split('\\')[2])
            file_directory_gt = os.path.join(folder_directory_gt, img_path.split('\\')[-1])

            if not os.path.isdir(folder_directory_gt):
                os.makedirs(folder_directory_gt)
                logger.info(f"New directory created '{folder_directory_gt}'")

            cv2.imwrite(file_directory_gt ,gt_img)


    def _flip_breast(self, img):
        # Determine the orientation of the breast and flip the image if required
        rotation_threshold = np.median(img[:1000, :400])

        if rotation_threshold < 10:
            img = cv2.flip(img, 1)
        
        return img
    
    def _add_padding(self, img, path, ratio=1):
        length, width = img.shape[:2]
        if length / width > ratio:
            add_wid = round(length / ratio - width)
            pad = np.zeros((length, add_wid), dtype=img.dtype)
            if '_R_' in path:
                return np.concatenate((pad, img), axis=1)
            return np.concatenate((img, pad), axis=1)
        return img

    def upsample(self, image, export_upsampled, dir, type):
        # Compute the new size for upsampling
        height, width = image.shape[:2]
        new_height = int(height * self.scale_factor)
        new_width = int(width * self.scale_factor)

        # Perform upsampling using resize
        upsampled_image = cv2.resize(image.copy(), (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Export processed images
        if export_upsampled:     
            upsample_dir = f'../dataset/output/upsampled/{type}'
            upsample_img_dir = os.path.join(upsample_dir, dir.split('\\')[1])
            # print(upsample_img_dir)
            if not os.path.isdir(upsample_dir):
                os.makedirs(upsample_dir)
                logger.info(f"New directory created '{upsample_dir}'")

            cv2.imwrite(upsample_img_dir ,upsampled_image)

        return upsampled_image
    
    def save_image(self, image, save_directory, image_filename):
        try:
            # Create the save directory if it doesn't exist
            os.makedirs(save_directory, exist_ok=True)
            
            # Save the image to the specified directory
            save_path = os.path.join(save_directory, image_filename)
            cv2.imwrite(save_path, image)
            
            print(f"Image saved successfully to: {save_path}")
        except Exception as e:
            print(f"Error saving image: {str(e)}")

    def prepare_export(self, image):
        image = image.astype(np.float32)
        image_uint8 = cv2.convertScaleAbs(image / np.max(image) * 255)
        return cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        
    def fit(self, dataset_path, ground_truth_path, process_n = None, plot = False, export_processed = False):
        if isinstance(dataset_path, str):
            start_time = time.time()
            logger.info("Started processing pipeline.")

            full_path_dirs = glob.glob(dataset_path+"\\*.tif")

            if process_n is None:
                process_n = len(full_path_dirs)
            elif process_n == 0:
                logger.warning("The number of processed images process_n can't be 0. \
                               To process the entire dataset, remove the argument when calling the function.")
                return
                        
            for path in tqdm(full_path_dirs[:process_n]):
                gt_path = os.path.join(ground_truth_path ,path.split('\\')[-1]) 

                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                # self.save_image(img, r'../report/report_images/preprocessing/', 'original.tif')

                # print(img.shape, img.dtype)
                gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)


                self._gray_img = img
                self._thresholding_mask = self._threshold_mask(self._gray_img)
                # self.save_image(self._thresholding_mask, r'../report/report_images/preprocessing/', 'thresholding_mask.jpg')

                # print(self._thresholding_mask)
                self._contour_img, self._segmented_img, self._gt_img = self._find_contours_and_segment(self._gray_img, gt_img)
                # self.save_image(self._segmented_img, r'../report/report_images/preprocessing/', 'segmented_img.tif')

                # self._rescaled_img = self._rescale(self._segmented_img)
                self._clahe_img = self._clahe(self._segmented_img)
                # self.save_image(self._clahe_img, r'../report/report_images/preprocessing/', 'clahe_img.tif')

                self._resized_img, self._gt_img = self._resize(self._clahe_img, self._gt_img)
                
                # Convert resized_img to uint16
                self._resized_img = self._resized_img.astype(np.uint16)
                # self.save_image(self._resized_img, r'../report/report_images/preprocessing/', 'final_resized.tif')


                if(self._gt_img is not None):
                    self._gt_img = self._gt_img.astype(np.uint8)

                # print(self._resized_img.shape, self._resized_img.dtype)
                # print(self._gt_img.shape, self._gt_img.dtype)


                # Some loggers to help keep track of the process
                # logger.info(f"Original image shape: {img.shape}")
                # logger.info(f"Resized image shape: {self._resized_img.shape}.")
                # logger.info(f"Grayscale image shape: {self._gray_img.shape}")  
                # logger.info(f"Thresholded Mask Shape: {self._thresholding_mask.shape}")       
                # logger.info(f"Preprocessing pipeline complete.")       
                
                # Displaying some results for validation
                if plot:
                    imgs = {
                        f"Original {img.shape[0]}x{img.shape[1]}": img, 
                        "Threshold Mask": self._thresholding_mask, 
                        "Segmented image on thresh mask": self._segmented_img, 
                        "_clahe_img": self._clahe_img, 
                    #     "Rescaled 16-bit":rescaled_img,
                        # "_resized_img": self._resized_img,
                        # f"GT {gt_img.shape[0]}x{gt_img.shape[1]}": gt_img,
                        # f"GT Cropped {self._gt_img.shape[0]}x{self._gt_img.shape[1]}": self._gt_img,
                        f"Preprocessed {self._resized_img.shape[0]}x{self._resized_img.shape[1]}": self._resized_img
                    }

                    display.plot_figures(imgs, 1,5) 

                
                # Export processed images
                if export_processed:                
                    self._export_processed(dataset_path, ground_truth_path, self._resized_img, self._gt_img, path)

            logger.info(f"Finished processing {len(full_path_dirs)} files in approximately {(time.time() - start_time):.03f} seconds.")
        else:
            raise NotImplementedError("dataset_path must be a directory string to the dataset files. Other formats are not yet implemented.")

        return



