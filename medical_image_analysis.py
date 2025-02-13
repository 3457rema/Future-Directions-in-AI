import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Simulate medical image data (e.g., pixel values)
X = np.random.rand(100, 256)  # 100 images, 256 features each (pixel values)
y = np.random.randint(0, 2, 100)  # Binary labels (0: healthy, 1: tumor)

# Train a simple AI model using RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)  # Fit the model to the simulated data

# Simulate a new medical image (1 new image with 256 features)
new_image = np.random.rand(1, 256)

# Predict whether the new image contains a tumor or not
prediction = model.predict(new_image)

# Output the result
if prediction[0] == 1:
    print("Tumor detected!")
else:
    print("No tumor detected.")
