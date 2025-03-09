import pandas as pd
import numpy as np

# Number of samples
num_samples = 1000

# Generate random labels (assuming 10 clothing categories)
labels = np.random.randint(0, 10, size=(num_samples, 1))

# Generate random pixel values (28x28 = 784 pixels, grayscale from 0 to 255)
pixels = np.random.randint(0, 256, size=(num_samples, 784))

# Combine labels and pixels into a DataFrame
df = pd.DataFrame(np.hstack((labels, pixels)))

# Name the first column as 'label', rest as pixel_0, pixel_1, ..., pixel_783
df.columns = ['label'] + [f'pixel_{i}' for i in range(784)]

# Save to CSV
df.to_csv("data.csv", index=False)

print("data.csv created successfully!")
