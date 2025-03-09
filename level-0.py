import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data.csv')

# Check dataset shape
print("Dataset Shape:", df.shape)

# Display the first few rows of the dataset
print(df.head())

# Display a few images with labels
plt.figure(figsize=(10, 5))
for i in range(10):
    image = df.iloc[i, 1:].values.reshape(28, 28)  # Assuming 28x28 grayscale images
    label = df.iloc[i, 0]  # First column as label
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Verify grayscale format by checking pixel value range
sample_image = df.iloc[0, 1:].values.reshape(28, 28)
print("Pixel Values Range:", sample_image.min(), "-", sample_image.max())
