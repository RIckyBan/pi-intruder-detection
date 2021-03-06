import argparse
import cv2
import numpy as np

print(cv2.__file__)

classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

model = cv2.dnn.readNetFromTensorflow('models/ssd/frozen_inference_graph.pb',
                                      'models/ssd/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')


def detect_obj(image):
    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
    output = model.forward()
    return output[0, 0, :, :]


def draw_bb(image, detections):
    flag = False
    res = []
    image_height, image_width = image.shape[:2]
    for detection in detections:
        confidence = detection[2]
        if confidence > .75:
            idx = detection[1]
            class_name = classNames[idx]
            if class_name == "person":
                flag = True
            res.append((class_name, confidence))

            # print(" "+str(idx) + " " + str(confidence) + " " + class_name)

            axis = detection[3:7] * (image_width,
                                     image_height, image_width, image_height)

            (start_X, start_Y, end_X, end_Y) = axis.astype(np.int)[:4]

            cv2.rectangle(image, (start_X, start_Y),
                          (end_X, end_Y), (23, 230, 210), thickness=2)
            cv2.putText(image, class_name+" "+str(confidence), (start_X, start_Y),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))

    return image, res, flag


def main(args):
    image = cv2.imread(args.image)

    detections = detect_obj(image)
    print(detections)
    image = draw_bb(image, detections)

    cv2.imshow('image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="input image")
    args = ap.parse_args()

    main(args)
