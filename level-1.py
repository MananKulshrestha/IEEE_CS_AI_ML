import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data.csv')

# 1️⃣ Basic Info
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

# 2️⃣ Check for Missing Values
print("\nMissing Values:", df.isnull().sum().sum())

# 3️⃣ Summary Statistics for Pixel Values
print("\nSummary Statistics:")
print(df.describe())

# 4️⃣ Pixel Statistics Function
def pixel_statistics(df):
    """Compute statistics for pixel values in the dataset."""
    pixel_values = df.iloc[:, 1:].values.flatten()  # Ignore the first column (labels)
    stats = {
        "Mean": np.mean(pixel_values),
        "Standard Deviation": np.std(pixel_values),
        "Min": np.min(pixel_values),
        "Max": np.max(pixel_values),
    }
    return stats

# Print pixel statistics
stats = pixel_statistics(df)
print("\nPixel Value Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")

# 5️⃣ Display Sample Images from Different Labels
def display_sample_images(df, num_samples=5):
    plt.figure(figsize=(10, 10))

    unique_labels = sorted(df['label'].unique())  # Get unique categories
    for label in unique_labels:
        samples = df[df['label'] == label].iloc[:num_samples, 1:].values  # Get pixel data
        for i, sample in enumerate(samples):
            image = sample.reshape(28, 28)  # Reshape into 28x28 grid
            plt.subplot(len(unique_labels), num_samples, label * num_samples + i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Label: {label}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

display_sample_images(df)

# 6️⃣ Heatmap of First 20 Features (Optional)
plt.figure(figsize=(10, 8))
corr = df.iloc[:, 1:21].corr()  # Correlation of first 20 pixels
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap (Partial)')
plt.show()
