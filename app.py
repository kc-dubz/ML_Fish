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
    'Abramis brama',
    'Acipenseridae',
    'Anguilla anguilla',
    'Aspius aspius',
    'Barbus barbus',
    'Blicca bjoerkna',
    'Carassius carassius',
    'Carassius gibelio',
    'Ctenopharyngodon idella',
    'Cyprinus carpio',
    'Esox lucius',
    'Gasterosteus aculeatus',
    'Gobio gobio',
    'Gymnocephalus cernuus',
    'Lepomis gibbosus',
    'Leuciscus cephalus',
    'Leuciscus idus',
    'Leuciscus leuciscus',
    'Neogobius fluviatilis',
    'Neogobius kessleri',
    'Neogobius melanostomus',
    'Perca fluviatilis',
    'Rhodeus amarus',
    'Rutilus rutilus',
    'Salmo trutta subsp. fario',
    'Sander lucioperca',
    'Scardinius erythrophthalmus',
    'Silurus glanis',
    'Tinca tinca',
    'Vimba vimba'
]

#Image Preprocessing
IMG_SIZE = 224

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
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
        preds = model.predict(img)[0]

        top_3_idx = preds.argsort()[-3:][::-1]

        prediction = []
        for i in top_3_idx:
            prediction.append(
                f"{class_names[i]} ({preds[i] * 100:.2f}%)"
            )

        prediction = "\n".join(prediction)

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