'''
Created on 2020 M10 31

@author: vaibh
'''

import cv2

cap = cv2.VideoCapture(0)


cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    ret,frame=cap.read()
    frame = cv2.cvtColor(frame,0)
    detection =  cascade_classifier.detectMultiScale(frame)
    
    if (len(detection) > 0):
        (x,y,w,h) = detection[0]
        
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,128,0),5)
        ##frame = cv2.circle(frame, (x,y), 40, (128, 128, 128), 2 )
    
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()