import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Startup Debug
print(">>> app.py is starting")

#Enable Flask
app = Flask(__name__)

#Model Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fish_classifier.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print(">>> loading model...")
model = load_model(MODEL_PATH)
print(">>> model loaded successfully")

#Class Names
class_names = [
    "Salmon",
    "Tuna",
    "Trout",
    "Tilapia",
    "Catfish"
]

#Image Preprocessing
IMG_SIZE = 224

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # Save uploaded image
        upload_path = os.path.join("static", file.filename)
        file.save(upload_path)

        # Predict
        img = prepare_image(upload_path)
        preds = model.predict(img)

        predicted_class = class_names[np.argmax(preds)]
        confidence = round(np.max(preds) * 100, 2)

        prediction = f"{predicted_class} ({confidence}%)"

        return render_template(
            "index.html",
            prediction=prediction,
            image_path=upload_path
        )

    return render_template("index.html", prediction=prediction)

#Run Server
if __name__ == "__main__":
    print(">>> starting flask server")
    app.run(debug=True)