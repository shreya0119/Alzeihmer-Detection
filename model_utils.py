import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Ensure these match exactly what Colab prints for train_ds.class_names
CLASSES = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']


def predict_alzheimer(image_path, weights_path):
    # 1. Rebuild the "Skeleton" (The Brain structure)
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(4, activation='softmax')
    ])

    # 2. Pour the "Knowledge" (Weights) into the skeleton
    model.load_weights(weights_path)

    # 3. Process the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # Standardization: This helps handle images from Roboflow/Web
    img=img.astype('float32')/255.0


    img = np.expand_dims(img, axis=0)

    # 4. Get raw probabilities
    predictions = model.predict(img)[0]

    # 5. SENSITIVITY HACK:
    # If Moderate (Index 1) is > 10%, we report it to be safe.
    if predictions[1] > 0.10:
        label = CLASSES[1]
        score = predictions[1]
    # If Mild (Index 0) is > 15%, we report it
    elif predictions[0] > 0.15:
        label = CLASSES[0]
        score = predictions[0]
    else:
        # Otherwise, pick the highest one normally
        label = CLASSES[np.argmax(predictions)]
        score = np.max(predictions)

    return label, round(float(score) * 100, 2)