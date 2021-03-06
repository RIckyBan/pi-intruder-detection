import cv2

from base_camera import BaseCamera


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
            yield frame
