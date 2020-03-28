import cv2  # de-facto package for image processing
import argparse  # package for using command line arguments

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=False, help="path to video for face detection")
args = vars(ap.parse_args())

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

# To use a video file as input 
# cap = cv2.VideoCapture(args['image'])

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (212, 32, 212), 3)
    # Display
    cv2.imshow('Smile Please 2', img)
    # Stop if 'q' key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==ord('q'):
        break
# Release the VideoCapture object
cap.release()
#close all the opened windows
cv2.destroyAllWindows()
