from imutils import face_utils, translate, resize
import dlib
import cv2
import numpy as np

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

camera = cv2.VideoCapture(0)

while True:

    _, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eyes = np.zeros((480, 640, 3), dtype='uint8')
    eyes_mask = eyes.copy()
    eyes_mask = cv2.cvtColor(eyes_mask, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[36:42]
        rightEye = shape[42:48]

        cv2.fillPoly(eyes_mask, [leftEye], 255)
        cv2.fillPoly(eyes_mask, [rightEye], 255)

        for point in shape[36:48]:
            cv2.circle(image, tuple(point), 2, (0, 255, 0), -1)

    cv2.imshow("Output", eyes_mask)

    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break

cv2.destroyAllWindows()
camera.release()

