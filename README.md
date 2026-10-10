# Beyond the Peel

A classical computer vision and machine learning system that classifies bananas as **ripe** or **unripe** from a single photo, using dark-spot segmentation, contour-based features, and a Support Vector Machine. No deep learning is used.

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Related Work](#related-work)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Experimental Results](#experimental-results)
7. [Discussion](#discussion)
8. [Challenges and Limitations](#challenges-and-limitations)
9. [Future Work](#future-work)
10. [Conclusion](#conclusion)
11. [Getting Started](#getting-started)
12. [Project Structure](#project-structure)
13. [References](#references)

## Abstract

This project builds a computer vision pipeline that separates ripe bananas from unripe ones. Each image is resized, filtered, and segmented to find dark spots on the peel. Two numbers are then extracted from the segmentation mask, the **spot count** and the **total spot area**, and three classical classifiers (Logistic Regression, K-Nearest Neighbors, and a linear SVM) are trained on them. The trained SVM is wrapped in a Gradio web interface for live prediction.

## Introduction

Computer vision lets a computer interpret images. This project uses classical methods, meaning hand-chosen mathematical tools such as thresholding, morphology, and contour analysis, which give full control over how each image is processed.

### Motivation

Grocery stores and food warehouses handle thousands of pieces of fruit a day, and checking each one by hand is slow. An automated ripeness check is useful for food sorting and factory quality control, where speed matters.

### Problem Statement

The goal is to take a banana photo and decide which class it belongs to. The dataset images come from public sources and vary in lighting, angle, and background. The pipeline has to clean these images and turn each one into a small list of numbers that a machine learning model can classify.

### Project Contribution

- A complete image classification pipeline built from traditional tools instead of deep learning.
- Preprocessing (resizing, median filtering, HSV conversion), dark-spot segmentation, and contour-based feature extraction.
- A comparison of Logistic Regression, KNN, and SVM on the extracted features.
- A Gradio interface that runs the same pipeline on a new image.

## Related Work

Classical pipelines remain attractive for fruit quality inspection because they are fast, cheap to run, and easy to tune by hand. The studies below motivated the approach. This project does not use HOG or LBP, because banana spots are irregular and not geometrically sharp, so contour-based features were used instead.

| Study | Method | Key takeaway |
|---|---|---|
| Huang (2023) | HOG + SVM/KNN | SVM outperformed KNN on fruit outlines. |
| Miller (2024) | LBP + KNN | Texture helps find decay better than color alone. |
| TIJER (2025) | Sobel + HOG | A lightweight classical pipeline is fast enough for conveyor-belt sorting. |
| Review (2026) | HOG + multiclass SVM | Manual threshold tuning is cost-effective for small-scale users. |

<!-- TODO: add full citations for the four studies above, or remove this section. They are not in the original reference list. -->

## Dataset

The dataset was collected from public sources (a Roboflow dataset) and split into two folders by class:

```
DATASET/
├── ripe/      # ripe bananas with many dark spots
└── unripe/    # unripe bananas with few or no dark spots
```

- Blurry images and images where the banana was cut off were removed.
- The notebook reads `.jpg`, `.jpeg`, and `.png` files and skips any file OpenCV cannot read.
- The notebook output reports a feature dataset of **750 images** in total.
- The SVM confusion matrix shown below has 88 ripe and 62 unripe test samples (150 in total, a 20% split), so the two classes are not perfectly balanced.

## Methodology

![Pipeline overview](images/pipeline_overview.png)

### 1. Image Preprocessing

Every image is resized to **256×256** so measurements such as spot area are comparable across bananas. A **5×5 median filter** removes noise, and the image is converted from BGR to **HSV**. Dark spots are isolated using the Value channel, which is less affected by the peel's green or yellow color.

![Preprocessing](images/preprocessing.png)

### 2. Filtering and Histogram Analysis

Gaussian filtering, grayscale histograms, and histogram equalization are used in the notebook to inspect image quality and contrast. They are analysis and visualization steps and are **not** part of the feature-extraction pipeline.

![Filtering and histograms](images/filtering_histograms.png)

### 3. Segmentation

The pipeline that produces the classification features is:

1. Median blur (5×5)
2. HSV conversion
3. `cv2.inRange` with `lower = [0, 0, 0]` and `upper = [179, 255, 60]`, which keeps pixels whose Value is 60 or lower
4. Erosion (3×3 kernel, 1 iteration) to remove small artifacts
5. Dilation (3×3 kernel, 3 iterations) to restore spot size

The same function (`segment_spots`) is used for both classes, for training, and for the web interface.

The notebook also demonstrates Otsu thresholding, morphological closing/opening, and largest-contour extraction on ripe images. This is an exploration of banana-versus-background separation and is **not** applied when building the features.

![Segmentation](images/segmentation.png)

### 4. Feature Extraction

Contours are found on each binary mask with `cv2.findContours`, and two features are computed:

- **Spot_Count**: the number of contours
- **Total_Area**: the sum of `cv2.contourArea` over all contours

Each image becomes one row in a Pandas DataFrame with columns `Spot_Count`, `Total_Area`, and `Class_Label`.

![Feature extraction](images/feature_extraction.png)

### 5. Machine Learning Classification

- **Split:** 80% training, 20% testing (`random_state=42`).
- **Scaling:** `StandardScaler`, since `Total_Area` is in the thousands while `Spot_Count` is usually small.
- **Models:** Logistic Regression, KNN (`n_neighbors=3`), and SVM (`kernel='linear'`).
- **Metric:** classification accuracy on the held-out test set, plus a confusion matrix for the SVM.

The notebook also trains a second SVM directly on the flattened 256×256 masks as a comparison against the two-feature approach.

## Experimental Results

Accuracy from the original notebook run:

| Model | Accuracy |
|---|---|
| Logistic Regression | 80.00% |
| K-Nearest Neighbors (k=3) | 83.33% |
| SVM (linear kernel) | 82.00% |

![Model accuracy output](images/model_accuracy.png)

SVM confusion matrix and feature importances (absolute weights of the scaled features):

![Confusion matrix](images/confusion_matrix.png)

![Feature importance](images/feature_importance.png)

> **Note:** The confusion matrix above comes from a different run than the accuracy table. It gives (65 + 47) / 150 ≈ 74.7% for the SVM, not 82%. The segmentation code was also unified after these figures were produced. Re-run the notebook to regenerate every number and figure from the current code before citing them.

## Discussion

- In the accuracy table, **KNN scored highest (83.33%)**, followed by the SVM (82.00%) and Logistic Regression (80.00%). The three models are within a few points of each other.
- The SVM uses a **linear** kernel, so it finds a straight decision boundary in the scaled feature space.
- The feature-importance chart shows that **Spot Count** carries much more weight (about 1.6) than **Total Area** (about 0.3).
- Segmentation quality is the main driver of results. Isolating dark spots with the HSV Value channel (`upper = [179, 255, 60]`) is what turns the images into usable numbers.

## Challenges and Limitations

- **Dataset diversity:** images vary in noise, dust, lighting, and angle.
- **Shadows and dark backgrounds:** the spot mask is not restricted to the banana, so shadows and dark background pixels can be counted as spots and distort `Total_Area`.
- **Limited features:** only two features are used. A banana that turns uniformly brown without distinct spots, or whose rot is on the side facing away from the camera, will be hard to classify.
- **Fixed thresholds:** the HSV threshold is hand-tuned and may need adjusting for different lighting.

## Future Work

- Restrict spot detection to the banana region (for example with the Otsu/contour mask) so shadows and backgrounds are ignored.
- Add color features such as yellow versus green peel.
- Try deep learning to avoid manual threshold tuning.
- Run the pipeline on live video, for example a camera above a conveyor belt.

## Conclusion

This project built a classical computer vision system that classifies bananas as ripe or unripe. Images are resized, median-filtered, and converted to HSV; dark spots are segmented with a Value threshold and cleaned with erosion and dilation; the spot count and total spot area are extracted into a Pandas DataFrame, scaled with `StandardScaler`, and used to train three classifiers. The results show that the quality of preprocessing and segmentation largely determines how well simple models perform.

## Getting Started

The notebook was written for **Google Colab** with the dataset stored on Google Drive.

1. Upload the dataset to Google Drive so the paths match:
   - `/content/drive/MyDrive/Project/DATASET/ripe`
   - `/content/drive/MyDrive/Project/DATASET/unripe`
2. Open a Colab notebook and install the dependencies if needed:
   ```bash
   pip install opencv-python numpy pandas matplotlib seaborn scikit-learn gradio
   ```
3. Copy `grandproject.py` into the notebook (or run it cell by cell) and execute it. Colab will ask for Drive access.
4. At the end, a Gradio interface launches. Upload or capture a banana photo to see the segmentation mask, the predicted class, and the extracted features.

To run outside Colab, remove the `google.colab` import and `drive.mount(...)` line, and change `UNRIPE_DIR` and `RIPE_DIR` to local paths.

## Project Structure

```
.
├── grandproject.py      # Full pipeline: preprocessing, features, training, GUI
├── README.md
└── images/              # Figures used in this README
```

## References

1. Bradski, G., & Kaehler, A. (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 51–56.
4. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362.
5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.
6. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
7. Suzuki, S. (1985). Topological structural analysis of digitized binary images by border following. *Computer Vision, Graphics, and Image Processing*, 30(1), 32–46.
8. Scikit-learn documentation. Logistic Regression, SVC, and KNeighborsClassifier module references.
