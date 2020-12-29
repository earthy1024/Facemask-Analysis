import numpy as np
import cv2
from datetime import datetime
import TrainingModel
from PIL import Image

model = TrainingModel.createModel()

cv2.namedWindow("preview")

vc = cv2.VideoCapture(0)
cv2.resizeWindow("preview", 256, 256)


if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False


good = "Thank you"
bad = "Please wear facemask properly"
worse = "Please put on facemask"
maskOn = True
while cv2.getWindowProperty("preview", 0) >= 0:

    success, image = vc.read()

    cv2.imwrite("frame.jpg", image)
    im = Image.open("frame.jpg")
    im = im.crop((192, 112, 448, 368))
    classification = TrainingModel.getClassification(im, model)
    if classification == "Masked":
        cv2.putText(image, good, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
    if classification == "Poorly Masked":
        cv2.putText(image, bad, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
    if classification == "Unmasked":
        cv2.putText(image, worse, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow("preview", image)


    key = cv2.waitKey(20)
    if key >= 0:
        break

cv2.destroyWindow("preview")