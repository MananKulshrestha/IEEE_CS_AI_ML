import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the images folder
folder_path = "C:/Users/Manan Kulshrestha/PycharmProjects/IEEE_CS_AI_ML/images/"

# List of image filenames
image_files = ["image1.png", "image2.png", "image3.png"]

# Load images using cv2, ensuring non-null values
images = [cv2.imread(folder_path + img, cv2.IMREAD_GRAYSCALE) for img in image_files]
images = [img for img in images if img is not None]  # Remove any failed loads

# Check if images were loaded
if not images:
    print("No valid images found. Please check the image path.")
    exit()

# Check dataset shape
print("Number of images:", len(images))
print("Shape of first image:", images[0].shape)

# Display images with labels
plt.figure(figsize=(10, 4))
for i, img in enumerate(images):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Image {i+1}")
    plt.axis("off")

plt.show()

# Verify grayscale format of the first image
if len(images[0].shape) == 2:
    print("The image is in grayscale format.")
else:
    print("The image is not in grayscale format.")
