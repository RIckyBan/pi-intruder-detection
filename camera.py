import cv2
from base_camera import BaseCamera
# from detect import detect_face, draw_bb
from ssd import detect_obj, draw_bb

cfgs = dict()
cfgs["mode"] = "haar"


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
            # faces = detect_face(cfgs, frame)
            # print(frame)
            detections = detect_obj(frame)
            if len(detections):
                frame = draw_bb(frame, detections)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
