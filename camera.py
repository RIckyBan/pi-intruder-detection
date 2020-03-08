import cv2
import requests

from base_camera import BaseCamera
# from detect import detect_face, draw_bb
from ssd import detect_obj, draw_bb

with open("token.txt") as f:
    token = f.read().strip()

IMAGE_NAME = "tmp.jpg"
cfgs = dict()
cfgs["mode"] = "haar"

def send_image():
    payload = {'message': '物体検知しました'}  # 送信メッセージ
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': 'Bearer ' + token}

    files={"image":open(IMAGE_NAME,"rb")}
    requests.post(url, data=payload, headers=headers,files=files,)  # LINE NotifyへPOST


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
            if len(detections):
                frame = draw_bb(frame, detections)
                cv2.imwrite(IMAGE_NAME, frame)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
