import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Determine absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the path to the "images" folder
image_folder = os.path.join(script_dir, "images")

# 3. List your image files here
image_files = [
    os.path.join(image_folder, "image1.png"),
    os.path.join(image_folder, "image2.png"),
    os.path.join(image_folder, "image3.png")
]

images_data = []
labels = []

# 4. Load images in grayscale;  skip any that fail
for file in image_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading {file}. Check if the file exists or is corrupted.")
        continue
    images_data.append(img)
    labels.append(os.path.basename(file))  # Just store the filename as label

# 5. If no images were loaded successfully, stop here
if not images_data:
    print("No valid images loaded. Please check file paths and names.")
    exit()

# 6. Check if all images have the same shape; if not, resize them
shapes = [img.shape for img in images_data]
if len(set(shapes)) > 1:
    print("Images have different shapes. Resizing them to match the first image's size...")
    target_shape = shapes[0]  # (height, width) of the first loaded image
    resized_data = []
    for img in images_data:
        resized = cv2.resize(img, (target_shape[1], target_shape[0]))  # Note: width, height order
        resized_data.append(resized)
    images_data = resized_data

# 7. Convert list of images to a 3D NumPy array (num_images, height, width)
images_data = np.array(images_data)
print("Dataset shape:", images_data.shape)

# 8. Display all loaded images
fig, axes = plt.subplots(1, len(images_data), figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(images_data[i], cmap="gray")
    ax.set_title(labels[i])
    ax.axis("off")
plt.show()

# 9. Verify grayscale format
print("Single image shape:", images_data[0].shape)
print("Pixel Value Range:", images_data.min(), "to", images_data.max())
