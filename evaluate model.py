import cv2
# import os
#print(os.environ["PATH"])

import numpy as np
from tensorflow.keras.models import load_model
model = load_model("cnn_model.h5")


# Load the new image
img = cv2.imread("IMG_20201223_093518.jpg")  # Replace with your image path
img = cv2.resize(img, (32, 32))    # Resize to match the input size of the model
img = img.astype("float32") / 255  # Normalize pixel values (0-1)
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(img)
predicted_class = np.argmax(predictions)
print(f"Predicted Class: {predicted_class}")
