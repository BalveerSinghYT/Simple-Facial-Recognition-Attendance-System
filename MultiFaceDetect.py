import cv2
import face_recognition
import os
import numpy as np

camera_ip = 'http://172.16.2.107:8080/video'
cv2.imread('ImageBase')
path = 'ImageBase'
cap = cv2.VideoCapture(camera_ip)
images = []
image_name = []

def names():
    images_list = os.listdir(path)
    for cl in images_list:
        curImage = cv2.imread(f'{path}/{cl}') 
        images.append(curImage)
        image_name.append(os.path.splitext(cl)[0])
        print(image_name)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

names()
encodeListKnown = findEncodings(images)
print("Encoding Complete")


while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, face_loc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) 
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(face_loc)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = image_name[matchIndex].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.rectangle(img, (x1, y2-35), (y2, y2),(0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
