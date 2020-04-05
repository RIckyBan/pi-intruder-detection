import cv2
import datetime
import os
import shutil
import time as t
from importlib import import_module
from flask import Flask, render_template, Response

import tensorflow as tf
from mtcnn.mtcnn import MTCNN

from camera import Camera
from utils import send_image
from detect import detect_face, draw_bb
# from ssd import detect_obj, draw_bb

app = Flask(__name__)
detector = MTCNN()
graph = tf.compat.v1.get_default_graph()

cfgs = dict()
cfgs["mode"] = "lbp"

@app.route('/')
def index():
    return "Hello World"

@app.route('/stream')
def stream():
    """Video streaming home page."""
    return render_template('stream.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        # # frame = cv2.flip(frame, 0)

        time = datetime.datetime.now()
        strtime = time.strftime('%Y%m%d-%H-%M-%S')
        IMG_NAME = strtime + ".jpg"
        IMG_PATH = "./img/raw/" + IMG_NAME

        cv2.imwrite("tmp.jpg", frame)
        detections = detect_face(detector, cfgs, frame)
        # # detections = detect_obj(frame)
        frame, res, flag = draw_bb(cfgs, frame, detections)        

        if flag:
            shutil.copy("tmp.jpg", IMG_PATH)
            IMG_PATH = "./img/res/" + IMG_NAME
            cv2.imwrite(IMG_PATH, frame)
            send_image(IMG_PATH, res)
            t.sleep(3)

        # encode as a jpeg image and return it
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

@app.route('/chart')
def chart():
    legend = 'Monthly Data'
    labels = ["January", "February", "March", "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template('chart.html', values=values, labels=labels, legend=legend)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
