from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model("Blood_Cell.keras")

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Helper function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(img_path)

            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            classes = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
            prediction = model.predict(img_array)
            predicted_class = classes[np.argmax(prediction)]

            prediction = f"Predicted Blood Cell Type: {predicted_class}"
            image_path = img_path
        else:
            prediction = "⚠️ Please upload a valid image file (.jpg, .jpeg, .png)"

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
