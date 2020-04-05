import argparse
import cv2
import datetime
import numpy as np
import os

INTERVAL = 15

def detect_face(cfgs, img):
    # カスケードファイルのパス
    # if cfgs["mode"] == "haar":
    #    CASCADE_PATH = './haarcascade_frontalface.xml'
    # else:
    #    CASCADE_PATH = './lbpcascade_frontalface_improved.xml'
    
    CASCADE_PATH = './haarcascade_frontalface.xml'

    # 画像をグレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # カスケード分類器の読み込み
    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # 顔検出の実行: 人数分のBBの配列を出力
    faces = cascade.detectMultiScale(gray)

    return faces

def draw_bb(frame, faces):
    if len(faces):
        flag = True
    else:
        flag = False

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame, flag

def main(args):
    now = datetime.datetime(year=2019, month=12, day=31) # 初期化

    cfgs = dict()
    cfgs["mode"] = args.mode

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        t = datetime.datetime.now()
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.flip(frame, 0)

        meanPixelVal = np.mean(frame)
#        if meanPixelVal < 20:
#            frame += 85 - int(meanPixelVal)
#            meanPixelVal = np.mean(frame)
        
        # print("Average Pixel Value:", meanPixelVal)

        faces = detect_face(cfgs, frame)

        # 検出結果の可視化
        cv2_im = frame.copy()
        if len(faces):
            cv2_im = draw_bb(frame, faces)
            cv2.imwrite(os.path.join('static/images', "{0:%Y-%m-%d-%H-%M-%S}.png".format(t)), frame)

        cv2.imshow('frame', cv2_im)
        if (t - now).seconds/60 > INTERVAL:
            now = t
            cv2.imwrite(os.path.join('static/images', "{0:%Y-%m-%d-%H-%M-%S}.png".format(now)), cv2_im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", default="haar",
        help="detection mode")
    args = ap.parse_args()

    main(args)
