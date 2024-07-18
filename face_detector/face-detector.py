import cv2
from random import randrange

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam=cv2.VideoCapture(0)

while True:
    suc_frame_read,frame=webcam.read()
    
    grayscaled_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coordinates=trained_face_data.detectMultiScale(grayscaled_image)

    for (x,y,w,h) in face_coordinates:
        """ cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2) """
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 255, 255),2)

    cv2.imshow('WEBCAM',frame)
    key=cv2.waitKey(1)

    if key==32:
        break

webcam.release()
                                        
""" img = cv2.imread('rdj.jpg')

grayscaled_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_coordinates=trained_face_data.detectMultiScale(grayscaled_image)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('RDJ IMAGE',img)
cv2.waitKey() """

print('code completed')