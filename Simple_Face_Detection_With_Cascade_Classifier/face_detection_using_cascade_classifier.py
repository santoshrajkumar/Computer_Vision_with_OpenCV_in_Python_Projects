import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

#importing the test image
img = imread('test3.jpg')

# OpenCV provides the CascadeClassifier class that can be used to create a cascade classifier for face detection. The constructor can take a filename as an argument that specifies the XML file for a pre-trained model.

#initiating the classifier class
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

#This function will return a list of bounding boxes for all faces detected in the photograph
#Each box lists the x and y coordinates for the bottom-left-hand-corner of the bounding box, as well as the width and the height.rectangle for each box directly over the pixels of the loaded image using the rectangle() function that takes two points.

bbox = classifier.detectMultiScale(img)
for box in bbox:
  x, y, width, height = box
  x2, y2 = x + width, y + height
  rectangle(img, (x,y), (x2,y2), (0,0,255), 1)

#shows the image with detected face/s
imshow(img)
  
