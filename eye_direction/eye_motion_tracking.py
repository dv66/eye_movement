import cv2
import numpy as np
import time

_VIDEO_FILE_NAME = ["eye_recording.flv", 0]
cap = cv2.VideoCapture(_VIDEO_FILE_NAME[1])




'''
    Tunable Parameters
'''
_RIGHT_THRESHOLD = -50
_LEFT_THRESHOLD = 50
_UP_THRESHOLD = 75
_DOWN_THRESHOLD = -75
_MID_POINT_THRESHOLD = 100

_STATE_CHANGE_TIME_INTERVAL = 1   # in seconds
current_state = None
previous_midpoint = None
previous_timestamp = time.time()
complete_cycle = [0,0,0,0]



while True:
    current_timestamp =time.time()
    time_difference = current_timestamp - previous_timestamp
    ret, frame = cap.read()
    if ret is False:
        break
    # roi = frame[269: 795, 537: 1416]
    roi = frame[30: 800, 60: 1500]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if previous_midpoint == None:
            previous_midpoint = (x + int(w/2), y + int(h/2))
        current_midpoint = (x + int(w/2), y + int(h/2))
        if abs(current_midpoint[0]-previous_midpoint[0]) > _MID_POINT_THRESHOLD and abs(current_midpoint[1]-previous_midpoint[1]) > _MID_POINT_THRESHOLD:
            previous_midpoint = current_midpoint
            # print('<-- Midpoint shifted -->')
        horizontal_displacement = (x - previous_midpoint[0])
        vertical_displacement = (y - previous_midpoint[1])
        # DEBUG = f' => {horizontal_displacement}, {vertical_displacement}, {previous_midpoint}, [{x}, {y}]'
        DEBUG = ""
        if horizontal_displacement>0 and horizontal_displacement >= _LEFT_THRESHOLD and (not current_state == 'Left') and time_difference > _STATE_CHANGE_TIME_INTERVAL:
            current_state = 'Left'
            print('Left ' + DEBUG)
            previous_timestamp = current_timestamp
            complete_cycle[0] =1
        if horizontal_displacement<0 and horizontal_displacement <= _RIGHT_THRESHOLD and (not current_state == 'Right') and time_difference > _STATE_CHANGE_TIME_INTERVAL:
            current_state = 'Right'
            print('Right' + DEBUG)
            previous_timestamp = current_timestamp
            complete_cycle[1] = 1
        if vertical_displacement > 0 and vertical_displacement >= _UP_THRESHOLD and (
        not current_state == 'Up') and time_difference > _STATE_CHANGE_TIME_INTERVAL:
            current_state = 'Up'
            print('Up ' + DEBUG)
            previous_timestamp = current_timestamp
            complete_cycle[2] = 1
        if vertical_displacement < 0 and vertical_displacement <= _DOWN_THRESHOLD and (
        not current_state == 'Down') and time_difference > _STATE_CHANGE_TIME_INTERVAL:
            current_state = 'Down'
            print('Down' + DEBUG)
            previous_timestamp = current_timestamp
            complete_cycle[3] = 1
        cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)

    if complete_cycle == [1,1,1,1] :
        print('Left, Right, Up, Down Done !!')
        complete_cycle = [0,0,0,0]
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
