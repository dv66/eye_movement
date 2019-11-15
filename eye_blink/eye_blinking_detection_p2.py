import cv2
import numpy as np
import dlib
from math import hypot
import time
import sys


cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

'''
Constants
'''
_BLINKING_RATIO_THRESHOLD = 5.7
_MIN_TIME_INTERVAL_THRESHOLD = 0.5                  # in seconds
_MAX_TIME_INTERVAL_THRESHOLD = 3
index = 0
previous_time_stamp = time.time()



def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    try:
        ratio = hor_line_lenght / ver_line_lenght
    except Exception as e:
        ratio = 1000000
    return ratio



while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > _BLINKING_RATIO_THRESHOLD:
            current_time_stamp = time.time()
            time_difference = current_time_stamp - previous_time_stamp
            if time_difference >= _MIN_TIME_INTERVAL_THRESHOLD and time_difference <= _MAX_TIME_INTERVAL_THRESHOLD:
                print(f'You blinked {index} times : ')
                # cv2.putText(frame, "BLINKING", (50, 150), font, 7, (0, 0, 255))
                index+=1
            if time_difference > _MAX_TIME_INTERVAL_THRESHOLD:
                print("***************** Try Again! ****************")
                index=0
            previous_time_stamp = current_time_stamp
        if index == 6:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (0, 0, 255))
            print("***************** YOU BLINKED  ***************")
            index = 0


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()