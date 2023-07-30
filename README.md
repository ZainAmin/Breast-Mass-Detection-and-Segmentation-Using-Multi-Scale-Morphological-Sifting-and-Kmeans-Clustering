# <h1 align="center">Breast Mass Detection and Segmentation Using Multi-scale Morphological Sifting and K-Means Clustering</h1>

![alt text](https://github.com/abdalrhmanu/mammographic-breast-mass-detection-and-segmentation/blob/main/report/report_images/visualize_segmentation/segmentation_results_cropped.png?raw=true)

Setup Environment Locally
============

To set up a virtual environment, follow the procedure found in <a href="https://github.com/abdalrhmanu/mammogram-mass-detection/blob/main/env.setup.md" target="_blank"> `env.setup.md`</a>.

Directory Structure
============

```
.
├── dataset             # Holds the project dataset folders and files.
    ├── output/features # Features extracted.
    ├── groundtruth     # Ground truth dataset files.
    ├── images          # Dataset images files.
    ├── masks           # Dataset masks files.
    ├── overlay         # Dataset overlay files.
    ├── all.txt         # All images names stored in a .txt file.
    ├── negatives.txt   # All negative labelled images names stored in a .txt file.
    └── positives.txt   # All positive labelled images names stored in a .txt file.
├── demonstration       # Demonestration file for testing the project.
├── literature          # Documentation/paper/project description, etc..
├── helpers             # Some developed packages and modules.
├── report              # Files and figures used to prepare the report.
├── submission reports  # Final submission files.
└── notebooks           # Jupyter notebooks used for development.

```

Project Methodology and Result
============
The proposed approach for mass detection and segmentation consists of five main steps for region segmentation, which are pre-processing, region candidate generation using multi-scale morphological sifting, mean shift filtering, k-means clustering, and finally post-processing. ucasML tool was used for obtaining a classification score using 10-fold cross validation for each region candidate for a binary classification task as it is robust to class imbalance. This project also includes feature extraction process and evaluation methods used.

<p align="center">
  <img src="https://github.com/abdalrhmanu/breast-mass-detection-and-segmentation-using-multi-scale-morphological-sifting-and-kmeans-clustering/blob/main/report/report_images/full-cropped.png?raw=true" alt="LaTeX Image">
</p>

This approach was implemented and evaluated on the InBreast mammographic dataset, which was able to segment mass lesions from the background accurately with a sensitivity of 76.92%. Various features were extracted from raw images and used in training ucasML tool. ucasML tool was able to provide satisfactory scores for both labels based on the features trained with while facing a high class imbalance. In general, the proposed system achieves promising results in both detection and segmentation of mammogram masses.

![alt text](https://github.com/abdalrhmanu/breast-mass-detection-and-segmentation-using-multi-scale-morphological-sifting-and-kmeans-clustering/blob/main/report/report_images/combined_froc.jpg?raw=true)







