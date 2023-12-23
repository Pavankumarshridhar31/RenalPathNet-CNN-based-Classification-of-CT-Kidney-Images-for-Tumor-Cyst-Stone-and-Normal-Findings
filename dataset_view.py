# -*- coding: utf-8 -*-
"""kideney_stone_prediction.ipynb


! unzip ct-kidney-dataset-normal-cyst-tumor-and-stone.zip

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import keras
import cv2
import seaborn as sns
from tensorflow.keras.utils import load_img
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage import transform
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns

kidney_data_df = pd.read_csv("/content/kidneyData.csv", header=0)

kidney_data_df.head()

kidney_data_df.tail()

kidney_data_df.dtypes

kidney_data_df.info()

# Display unique values in the 'Class' column
unique_classes = kidney_data_df['Class'].unique()

# Count occurrences of each class
kidney_classes = kidney_data_df['Class'].value_counts()

# Calculate the total number of elements
sum_of_elements = kidney_classes.sum()

# Print unique classes
print("Unique kidney classes:\n", unique_classes)

# Print class counts
print("Kidney classes counts:\n", kidney_classes)

# Print the total number of elements
print("Sum of elements is:", sum_of_elements)

import seaborn as sns
import matplotlib.pyplot as plt



# Define custom colors for each class
colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange']

# Create a count plot with custom colors
sns.countplot(x='Class', data=kidney_data_df, palette=colors)

# Set plot title and axis labels
plt.title('Number of Cases', fontsize=14)
plt.xlabel('Case Type', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Show the plot
plt.show()

normal_ct_path = "/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Normal"
cyst_ct_path = "/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Cyst"
stone_ct_path = "/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Stone"
tumor_ct_path = "/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor"

def plot_df_images(item_dir, num_imgs=2):
    # List all items (files) in the specified directory
    all_item_dirs = os.listdir(item_dir)

    # Select a subset of files (images) to display
    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]

    # Set up a 1xnum_imgs grid for subplots
    plt.figure(figsize=(15, 5))

    # Iterate through the selected images
    for idx, img_path in enumerate(item_files):
        plt.subplot(1, num_imgs, idx+1)  # Subplot index starts from 1
        img = plt.imread(img_path)
        plt.title(f'Image {idx+1}\nLabel: {img_path[-10:-4]}')  # Display image index and label in the title
        plt.imshow(img, extent=[0, 100, 0, 100])  # Adjust extent to set a common size for all images

    plt.tight_layout()

# Example usage:
print("Normal Samples:")
plot_df_images(normal_ct_path, num_imgs=3)
plt.show()

print("\nCyst Samples:")
plot_df_images(cyst_ct_path, num_imgs=3)
plt.show()

print("\nStone Samples:")
plot_df_images(stone_ct_path, num_imgs=3)
plt.show()

print("\nTumor Samples:")
plot_df_images(tumor_ct_path, num_imgs=3)
plt.show()

import os

# Specify the directory paths
normal_ct_dir = "/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Normal"
cyst_ct_dir = "/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Cyst"
stone_ct_dir = "/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Stone"
tumor_ct_dir = "/content/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor"

# Get the paths of image files in each directory
normal_ct_paths = [os.path.join(normal_ct_dir, file) for file in os.listdir(normal_ct_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
cyst_ct_paths = [os.path.join(cyst_ct_dir, file) for file in os.listdir(cyst_ct_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
stone_ct_paths = [os.path.join(stone_ct_dir, file) for file in os.listdir(stone_ct_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
tumor_ct_paths = [os.path.join(tumor_ct_dir, file) for file in os.listdir(tumor_ct_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def plot_image_comparison(img1_path, img2_path, label1='Image 1', label2='Image 2'):
    try:
        print(f"Attempting to read images from paths: {img1_path} and {img2_path}")
        img1 = mpimg.imread(img1_path)
        img2 = mpimg.imread(img2_path)

        if img1 is not None and img2 is not None:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(img1)
            plt.title(label1)
            plt.xlabel(os.path.basename(img1_path))  # Display the image file name as xlabel

            plt.subplot(1, 2, 2)
            plt.imshow(img2)
            plt.title(label2)
            plt.xlabel(os.path.basename(img2_path))  # Display the image file name as xlabel

            plt.tight_layout()
            plt.show()
        else:
            print("Error: Unable to read one or both of the images")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
plot_image_comparison(normal_ct_paths[45], cyst_ct_paths[45], label1='Normal', label2='Cyst')
