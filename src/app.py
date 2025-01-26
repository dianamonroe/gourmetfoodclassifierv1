from flask import Flask, request, render_template
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO('./Yolov86thRoundbestWeights.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            filepath = f'./uploads/{uploaded_file.filename}'
            uploaded_file.save(filepath)
            results = model.predict(filepath)
            return render_template('index.html', predictions=results)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
