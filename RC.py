import os
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from google.colab import drive
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

drive.mount('/content/drive')

UNRIPE_DIR = "/content/drive/MyDrive/Project/DATASET/unripe"
RIPE_DIR = "/content/drive/MyDrive/Project/DATASET/ripe"

SUPPORTED = (".jpg", ".jpeg", ".png")
IMG_SIZE = (256, 256)
KERNEL = np.ones((3, 3), np.uint8)
SPOT_LOWER = np.array([0, 0, 0])
SPOT_UPPER = np.array([179, 255, 60])


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED))


def segment_spots(img_bgr):
    resized = cv2.resize(img_bgr, IMG_SIZE)
    filtered = cv2.medianBlur(resized, 5)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SPOT_LOWER, SPOT_UPPER)
    eroded = cv2.erode(mask, KERNEL, iterations=1)
    return cv2.dilate(eroded, KERNEL, iterations=3)


def load_class(folder):
    files, masks, skipped = [], [], 0
    for fname in list_images(folder):
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            skipped += 1
            print("skipped (cannot read):", fname)
            continue
        files.append(fname)
        masks.append(segment_spots(img))
    return files, masks, skipped


def pick_random_image(folder, files):
    fname = random.choice(files)
    img = cv2.imread(os.path.join(folder, fname))
    return fname, cv2.resize(img, IMG_SIZE)


ripe_files = list_images(RIPE_DIR)
unripe_files = list_images(UNRIPE_DIR)
print("Found ripe images:", len(ripe_files))
print("Found unripe images:", len(unripe_files))

ripe_files, ripe_masks, ripe_skipped = load_class(RIPE_DIR)
print("Ripe skipped:", ripe_skipped)

unripe_files, unripe_masks, unripe_skipped = load_class(UNRIPE_DIR)
print("Unripe skipped:", unripe_skipped)


def show_filtering_and_histograms(folder, files, name, colors):
    fname, img = pick_random_image(folder, files)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title(f"Original {name} Grayscale\n({fname})")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title("1. Gaussian Filtered\n(Noise Reduction)")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(equalized, cmap='gray')
    plt.title("2. Histogram Equalized\n(Contrast Enhancement)")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.hist(gray.ravel(), 256, [0, 256], color='dimgray', alpha=0.8)
    plt.title("Original Grayscale Histogram")
    plt.xlabel("Pixel Intensity (0 - 255)")
    plt.ylabel("Pixel Count")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.hist(blurred.ravel(), 256, [0, 256], color=colors[0], alpha=0.8)
    plt.title("Gaussian Blurred Histogram")
    plt.xlabel("Pixel Intensity (0 - 255)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    plt.hist(equalized.ravel(), 256, [0, 256], color=colors[1], alpha=0.8)
    plt.title("Equalized Histogram")
    plt.xlabel("Pixel Intensity (0 - 255)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def show_output_check(folder, files, masks):
    idx = random.randint(0, len(files) - 1)
    fname = files[idx]
    original_bgr = cv2.imread(os.path.join(folder, fname))
    original_rgb = cv2.cvtColor(cv2.resize(original_bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title(f"Original: {fname}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Input")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(masks[idx], cmap='gray')
    plt.title("Detected Dark Spots (White)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


show_filtering_and_histograms(RIPE_DIR, ripe_files, "Ripe", ('royalblue', 'crimson'))

fname, img_resized = pick_random_image(RIPE_DIR, ripe_files)
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh_val, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
morph_closed = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, KERNEL)
morph_cleaned = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, KERNEL)
contours, _ = cv2.findContours(morph_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
final_contour_mask = np.zeros_like(morph_cleaned)

if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(final_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
else:
    print("Warning: No contours detected!")

plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.title(f"Original Ripe Image ({fname})")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(otsu_mask, cmap='gray')
plt.title(f"1. Otsu's Thresholding\n(Calculated Threshold: {int(thresh_val)})")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(morph_cleaned, cmap='gray')
plt.title("2. Morphological Operations\n(Closed Holes & Opened Noise)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(final_contour_mask, cmap='gray')
plt.title("3. Contour Extraction\n(Largest Object Mask)")
plt.axis('off')

plt.tight_layout()
plt.show()

show_output_check(RIPE_DIR, ripe_files, ripe_masks)

show_filtering_and_histograms(UNRIPE_DIR, unripe_files, "Unripe", ('forestgreen', 'darkorange'))

fname, img_resized = pick_random_image(UNRIPE_DIR, unripe_files)
img_filtered = cv2.medianBlur(img_resized, 5)
hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)
spot_mask = cv2.inRange(hsv, SPOT_LOWER, SPOT_UPPER)
eroded = cv2.erode(spot_mask, KERNEL, iterations=1)
final_result = cv2.dilate(eroded, KERNEL, iterations=3)

plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.title(f"Original Unripe Image\n({fname})")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
plt.title("1. Median Filtered\n(Noise Reduction)")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(spot_mask, cmap='gray')
plt.title("2. HSV Value Masking\n(Initial Segmentation)")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(eroded, cmap='gray')
plt.title("3. Morphological Erosion\n(Removing small artifacts)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(final_result, cmap='gray')
plt.title("4. Morphological Dilation\n(Final Segmented Mask)")
plt.axis('off')

plt.tight_layout()
plt.show()

show_output_check(UNRIPE_DIR, unripe_files, unripe_masks)

fname, img_resized = pick_random_image(UNRIPE_DIR, unripe_files)
gray_for_hist = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_for_hist, cmap='gray')
plt.title("Grayscale Banana")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.hist(gray_for_hist.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
plt.axvline(x=60, color='red', linestyle='dashed', linewidth=2, label='Spot Threshold (60)')
plt.title("Grayscale Histogram Visualization")
plt.xlabel("Pixel Intensity (0=Black, 255=White)")
plt.ylabel("Number of Pixels")
plt.legend()

plt.tight_layout()
plt.show()


def extract_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_count = len(contours)
    total_area = sum(cv2.contourArea(c) for c in contours)
    return spot_count, total_area


data_rows = []

for mask in unripe_masks:
    count, area = extract_features(mask)
    data_rows.append({'Spot_Count': count, 'Total_Area': area, 'Class_Label': 'Unripe'})

for mask in ripe_masks:
    count, area = extract_features(mask)
    data_rows.append({'Spot_Count': count, 'Total_Area': area, 'Class_Label': 'Ripe'})

feature_dataset = pd.DataFrame(data_rows)

print(f"Successfully created a dataset with {len(feature_dataset)} bananas!")
print("-" * 50)

display(feature_dataset.head(5))
display(feature_dataset.tail(5))

fname, original_img = pick_random_image(RIPE_DIR, ripe_files)
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

filtered_img = cv2.GaussianBlur(original_img, (5, 5), 0)
img_yuv = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2YUV)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
processed_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Before: Original Image")
plt.imshow(original_img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("After: Filtered & Histogram Equalized")
plt.imshow(processed_img)
plt.axis('off')

plt.tight_layout()
plt.show()

X = feature_dataset[['Spot_Count', 'Total_Area']]
y = feature_dataset['Class_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine (SVM)": SVC(kernel='linear', random_state=42)
}

print("--- MACHINE LEARNING CLASSIFICATION RESULTS ---\n")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

ripe_pixels = [m.flatten() for m in ripe_masks]
unripe_pixels = [m.flatten() for m in unripe_masks]

X_pix = np.array(ripe_pixels + unripe_pixels)
y_pix = np.array([1] * len(ripe_pixels) + [0] * len(unripe_pixels))

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pix, y_pix, test_size=0.2, random_state=42)

print(f"Total images processed: {len(X_pix)}")
print(f"Training on {len(Xp_train)} images, Testing on {len(Xp_test)} images")

pixel_svm = SVC(kernel='linear', random_state=42)
pixel_svm.fit(Xp_train, yp_train)
yp_pred = pixel_svm.predict(Xp_test)
print("Pixel-mask SVM Accuracy:", accuracy_score(yp_test, yp_pred))

cm_pix = confusion_matrix(yp_test, yp_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pix, display_labels=['Unripe', 'Ripe'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Greens')
plt.title('Confusion Matrix for Pixel-Mask SVM Model')
plt.show()

svm_model = models["Support Vector Machine (SVM)"]
y_pred = svm_model.predict(X_test_scaled)

print(f"Total samples processed: {len(X)}")
print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")
print(f"SVM Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
axes[0].set_title('Confusion Matrix (SVM Model)', fontsize=14, fontweight='bold', pad=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)

importances = np.abs(svm_model.coef_[0])
feature_names = ['Spot Count', 'Total Area']

sns.barplot(x=feature_names, y=importances, hue=feature_names, palette="Set1", legend=False, ax=axes[1])
axes[1].set_title('Feature Importances (Spot Count vs Total Area)', fontsize=14, fontweight='bold', pad=12)
axes[1].set_ylabel('Relative Importance (SVM Absolute Weight)', fontsize=12)
axes[1].set_xlabel('Extracted Features', fontsize=12)

plt.tight_layout()
plt.show()


def process_and_predict(input_image):
    if input_image is None or input_image.size == 0:
        return None, "Error", "No image detected. Please make sure to click 'Capture' before submitting!"

    if input_image.ndim == 2:
        input_bgr = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    elif input_image.shape[2] == 4:
        input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGBA2BGR)
    else:
        input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    final_mask = segment_spots(input_bgr)
    spot_count, total_area = extract_features(final_mask)

    features = pd.DataFrame([[spot_count, total_area]], columns=['Spot_Count', 'Total_Area'])
    features_scaled = scaler.transform(features)

    prediction = svm_model.predict(features_scaled)[0]
    debug_text = f"Raw: {spot_count} spots, {total_area} area | Scaled: {features_scaled[0]}"

    return final_mask, prediction, debug_text


interface = gr.Interface(
    fn=process_and_predict,
    inputs=gr.Image(),
    outputs=[
        gr.Image(label="Segmentation Mask"),
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Behind the Scenes (Features)")
    ],
    title="Banana Freshness Detector"
)

interface.launch(debug=True)
