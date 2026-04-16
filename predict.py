import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fish_classifier.keras"
DATA_DIR = BASE_DIR / "data"

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

class_names = sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()])

def predict_image(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    test_image = input("Enter image path: ")
    predict_image(test_image)