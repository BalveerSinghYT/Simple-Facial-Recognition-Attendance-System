import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# --*------ AutoMate using List ------------------
"""
Automatic find the images in folder & convert BGR to RGB, 
& encode it using the list.
"""
path = 'ImageBase'              # Folder Name
images = []                     #List with array of images
classNames = []                 #List with only images name text by spliting .jpg

mylist = os.listdir(path)       # Accessing all images in folder
print(mylist)

# Next we are going to use these images & import one by one

for cl in mylist:
    curImage = cv2.imread(f'{path}/{cl}')  # f'--' mean ImageAttendance/BillGates.jpg
    images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])   # remove .jpg extension
    #print(type(os.path.splitext(cl)))
print(classNames)

#--**---------- Function for find encodings ----------------------------------------
"""
    --->> BGR to RGB
    --->> Find encodings
    --->> append to encodeList & return encodeList
"""
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:         # r+ mean read & write
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtstring}')


#---------------------------------------------------------------------------------
encodeListKnown = findEncodings(images)
#print(len(encodeListKnown))
print("Encoding Complete")

#--***--------------- Taking test image using WebCam -----------------------------

cap = cv2.VideoCapture("http://25.76.62.158:8080/video")

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)     # Reducing size of webimage
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)        # Converting BGR to RGB
    """ 
        We may have multiple faces in webimage, so we will do following
        --->> Find all face_locations
        --->> Find encoding of all the faces
    """
    facesCurFrame = face_recognition.face_locations(imgS)           # This store coordinates of face location in list
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    #print(facesCurFrame)
    """Now we are going to iterate through all faces that we found in current frame
    & compare all these faces with the encoding that we found before"""

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):  # we r using zip to iterate on both at same time
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) # here we can give tolerance
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) # because we are sending in a list to face_distance funciton, it will return a list as well
        #print(faceDis)
        """
            Face Distance tell us the difference between known image & tested image.
            If the distance is <0.6, then matched       --- 0.6 is the default tolerance
            else if distance is >0.6, then did't match
                        &
            also we can change tolerance
        """
        # now we are going to use numpy to pick the minimum index value as match index
        matchIndex = np.argmin(faceDis)   # this gives the index no where the tolerance is minimum
        #print(matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            """
                Now the face recognition is done. Next step is to display the image window with rectangle on 
                face location & name of the person.
            """
            cv2.rectangle(img,(x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (y2, y2),(0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
            print(name, "is present")
            break
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
