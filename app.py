from flask import Flask, request
from ultralytics import YOLO
import os

app = Flask(__name__)

# تحميل المودل
model = YOLO("runs/detect/train/weights/best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return '''
    <h2>Upload Image for Detection</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <button type="submit">Upload</button>
    </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    results = model(filepath)

    results[0].save()

    return "Detection completed! Check runs/detect/predict"

import os

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)