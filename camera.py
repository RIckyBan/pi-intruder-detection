import cv2
import datetime
import requests
import time as t

from base_camera import BaseCamera
# from detect import detect_face, draw_bb
from ssd import detect_obj, draw_bb

with open("token.txt") as f:
    token = f.read().strip()

cfgs = dict()
cfgs["mode"] = "haar"

def send_image(IMG_PATH, res):
    print(IMG_PATH)
    message = '物体を検知しました\n\n'
    for obj in res:
        class_name, val = obj
        message += class_name + " " + str(val) +"\n"
    payload = {'message': message}  # 送信メッセージ
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': 'Bearer ' + token}

    files = {'imageFile': open(IMG_PATH, 'rb')}
    r = requests.post(url, headers=headers, params=payload, files=files)  # LINE NotifyへPOST
    print(r.text)

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        super().__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, frame = camera.read()
            # frame = cv2.flip(frame, 0)
            # detections = detect_face(cfgs, frame)
            # print(frame)
            detections = detect_obj(frame)
            frame, res = draw_bb(frame, detections)
            if len(res):
                time = datetime.datetime.now()
                strtime = time.strftime('%Y%m%d-%H-%M-%S')
                IMG_PATH = "./img/" + strtime + ".jpg"
                cv2.imwrite(IMG_PATH, frame)
                send_image(IMG_PATH, res)
                t.sleep(3)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
