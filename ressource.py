'''
this file will contain the class that will be using to read the file
'''
import cv2 as cv
import streamlit as st
import os
import numpy as np
from kalmanFilter import KalmanFilter



def face_recognition(frame) -> tuple :
    face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_scale , 1.1 , 4)
    if len(faces)>0:
        x , y , h , w = faces[0]
        result = [x+(h/2) , y+(w/2)]
        return result
    else :  return None

if __name__ == "__main__" :
    # VIDEO_PATH = 'video.mov'

    # if os.path.isfile(VIDEO_PATH) : video_cap = cv.VideoCapture(VIDEO_PATH)
    # else :
    #     raise FileNotFoundError

    # if not video_cap.isOpened() : raise SystemError

    video_cap = cv.VideoCapture(0)

    last_pos_X = [0]
    last_pos_Y = [0]

    frame_count = 0
    KF = KalmanFilter(0.1 , [0 , 0])
    while video_cap.isOpened() :
        ret , frame = video_cap.read()
        stat = KF.predict().astype(np.int32)
        print(stat)
        if ret :
            frame_count += 1
            face = face_recognition(frame)
            cv.circle(frame, (int(stat[0]) , int(stat[1])), 2, (0, 255, 0), 5)
            cv.arrowedLine(frame, 
                           (int(stat[0]), int(stat[1])),(int(stat[0]+stat[2]), int(stat[1]+stat[3])),
                           color=(0, 255, 0),
                           thickness=3,
                           tipLength=0.2
                           )
            if face != None :
                cv.circle(frame, (int(face[0]) , int(face[1])), 10 , (0, 0, 255), 2)
                KF.update(np.expand_dims(face, axis=-1))

            cv.imshow('ceve' ,frame)

            # TODO : use the numpy polynomial fit to find the curve that optimizes 
        
        
        else :
            break
        if cv.waitKey(25) & 0xFF == ord('q') :
            cv.destroyAllWindows()
            break






