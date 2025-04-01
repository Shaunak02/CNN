from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('cnn_cifar100_resnet50_model.h5')

# Load and preprocess a new image
image = cv2.imread('20201223_093518.jpg')
image = cv2.resize(image, (32, 32))  # Resize to match the CIFAR-100 input size
image = image.astype('float32') / 255.0  # Normalize to [0, 1]
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict the class of the new image
predictions = model.predict(image)

# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)[0]
print(f"Predicted Class: {predicted_class}")