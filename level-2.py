# level-2.py
# C:\Users\Manan Kulshrestha\PycharmProjects\IEEE_CS_AI_ML\level-2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import shap

# 1. Load and Inspect the Dataset
df = pd.read_csv('data.csv')
print("Dataset Shape:", df.shape)
print(df.head())

# 2. Dictionary to Map Label Numbers to Clothing Types
clothing_labels = {
    0: "T-shirt/Top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
}

# 3. Display Sample Images with Labels
plt.figure(figsize=(10, 10))
for label in range(10):
    images = df[df['label'] == label].iloc[:5, 1:].values
    for i in range(len(images)):
        plt.subplot(10, 5, label * 5 + i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(clothing_labels[label])
plt.tight_layout()
plt.show()

# 4. Data Preprocessing
X = df.drop('label', axis=1).values / 255.0
y = df['label'].values

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Logistic Regression Model
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_proba)

print("Accuracy:", accuracy)
print("Log Loss:", loss)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix Visualization
plt.figure(figsize=(10, 7))
sns.heatmap(
    confusion_matrix(y_test, y_pred), 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=[clothing_labels.get(i, str(i)) for i in sorted(set(y_test))],
    yticklabels=[clothing_labels.get(i, str(i)) for i in sorted(set(y_test))]
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 9. Explainable AI with SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance (Test Set)")
plt.tight_layout()
plt.show()
