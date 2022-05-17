from flask import Flask, render_template, request, Response, redirect, url_for
import os
from werkzeug.utils import secure_filename, send_from_directory, send_file
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def home():  # put application's code here
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def detect_images():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('static/',f.filename))
        ipath = os.path.join('static/',f.filename)
        # model = torch.load('runs/train/exp/weights/last.pt')
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/last.pt')
        result = model(ipath)
        result.render()
        # result.show()
        result.save(f.filename,'static/detect')
        img = os.path.join('static/detect/', f.filename)
        return render_template('index.html', filename=img)

@app.route('/<filename>')
def display_image(filename):
    return redirect(url_for(filename=filename), code=301)

# def gen():
#     """Video streaming generator function."""
#     cap = cv2.VideoCapture(0)
#     models = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/last.pt', force_reload=True)
#     while True:
#         frame = cap.read()
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#         else:
#             results = models(frame)
#             # cv2.imshow('YOLO', np.squeeze(results.render()))
#             buffer = cv2.imencode('YOLO', '.jpg', np.squeeze(results.render()))
#             frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
# @app.route('/detect_webcam')
# def detect_webcam():
#     return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    models = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/last.pt', force_reload=True)
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break
        results = models(frame)
        cv2.imwrite('demo.jpg', np.squeeze(results.render()))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

@app.route('/detect_webcam')
def detect_webcam():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
