from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO("Yolov86thRoundbestWeights.pt")  # Modelo entrenado

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(file.stream)
            results = model.predict(image, conf=0.25)
            label = results[0].names[results[0].boxes.cls[0].item()]
            confidence = results[0].boxes.conf[0].item() * 100
            return render_template(
                "index.html", label=label, confidence=confidence, image=file.filename
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
