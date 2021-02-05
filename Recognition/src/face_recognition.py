'''
Created on 2020 M11 1

@author: vaibh
'''
import cv2 #cv2 module imports OpenCV library
import sys #sys imports common python functions, such as argv
import os

imagePath = r'C:\Users\vaibh\OneDrive\Desktop\Projects\Recognition\src\1.jpg' #Pythonic way of reading the first argument is to assign the value returned by sys.argv[1]

#common practice is to convert the input image into gray scale for better results
image = cv2.imread(imagePath) #this function is used to take the input image and converts it into OpenCV object
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #.ctvColor function converts the input image object to grayscale object

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #faceCascade object that will load the Haar Cascade file with the cv2.CascadeClassifier method
faces = faceCascade.detectMultiScale(
    gray, #This specifies the use of the OpenCV grayscale image object that you loaded earlier.
    scaleFactor = 1.3, #This parameter specifies the rate to reduce the image size at each image scale.
    minNeighbors = 3, #This parameter specifies how many neighbors, or detections, each candidate rectangle should have 
                      #to retain it. A higher value may result in less false positives, but a value too high can eliminate true positives.
    minSize = (30,30) #This allows you to define the minimum possible object size measured in pixels. Objects smaller than this parameter are ignored.
    
) # .detectMultiScale() method on the facecascade object. This generates a list of rectangles for all of the detected faces in the image.
#The list of rectangles is a collection of pixel locations from the image, in the form of Rect(x,y,w,h).

print ("[INFO] Found {0} Faces!".format(len(faces))) #this prints out how many faces it detected

for (x,y,w,h) in faces: #for loop is used to iterate through the list of pixel location returned from faceCascade.detectMultiScale method
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) 
    roi_color = image[y:y + h, x:x + w]
    print ("[INFO] Object found.saving locally.")
    cv2.imwrite(str(x) + str(h) + '_face.jpg', roi_color)
    
    #image - tells code to draw rectangles on the original input image
    #(x,y), (x+w, y+h) - are the four pixel locations for the detected object. rectangle will use these to locate and draw rectangles around the detected objects in the input image.
    #(0, 255, 0) is the color of the shape. This argument gets passed as a tuple for BGR. 
    #2 is the thickness of the line measured in pixels.
    
#now we use .imwrite() method to write the new image to local filesystem as face_detected.jpg
path = r'C:\Users\vaibh\OneDrive\Desktop\Projects\Recognition\src\1.jpg'
directory = r'C:\Users\vaibh\OneDrive\Desktop\Projects\Recognition\src'
os.chdir(directory)
status = cv2.imwrite('face_detected.png', image) #This method will return true if the write was successful and false if it wasn’t able to write the new image.
print ("Image faces_detected.jpg written to filesystem: ",status)



