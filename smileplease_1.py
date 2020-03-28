import cv2 # de-facto package for image processing 
import argparse #package for using command line arguments

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to images for face detection")
args = vars(ap.parse_args())

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread(args['image'])
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (212, 32, 212),3)
#Parameters:
# input image 
# starting coordinates of the rectangle
# end coordinates of the rectangle
# color of the rectangle(I have used pink here): you can input RGB values for any color using
#  https://www.google.com/search?ei=HCt_Xt65OdfGrQHggaXgCQ&q=color+picker&oq=color+picker&gs_lcp=CgZwc3ktYWIQAzIECAAQQzIECAAQQzIECAAQQzICCAAyAggAMgIIADICCAAyAggAMgIIADICCAA6BAgAEEc6BQgAEIMBUOURWIIgYJIhaABwAXgAgAH6AogBmRGSAQcwLjkuMS4xmAEAoAEBqgEHZ3dzLXdpeg&sclient=psy-ab&ved=0ahUKEwjet-n8_rzoAhVXYysKHeBACZwQ4dUDCAs&uact=5
# thickness of the rectangle
# Display the output
cv2.imshow('Smile Please 1', img)
# Wait for key presses
k = cv2.waitKey(0)
# If Esc is pressed
if k == 27:
    #Path and name of the our image with the face recognized
    cv2.imwrite('output_images/face_recognized_1.png', img)
    #close all the opened windows
    cv2.destroyAllWindows()
