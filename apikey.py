import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier("C:\\Users\\saibalaji\\Downloads\\haarcascade_frontalface_default.xml")

# Load a sample image of the authorized user
authorized_user_image = cv2.imread("C:\\Users\\saibalaji\\Videos\\iVCam\\20240329205500.jpg", cv2.IMREAD_GRAYSCALE)

# Extract facial features from the authorized user's image
authorized_user_face = face_cascade.detectMultiScale(authorized_user_image, scaleFactor=1.1, minNeighbors=5)
print(authorized_user_face)

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract facial features from the detected face
        roi_gray = gray[y:y+h, x:x+w]
        
        # Perform face recognition by comparing with the authorized user's face
        # For simplicity, here we're just checking if the dimensions of the detected face match the authorized user's face
        if len(authorized_user_face) > 0 and (w+h) == (authorized_user_face[0][2]+authorized_user_face[0][3]):
            # Authorized user detected
            cv2.putText(frame, 'Authorized User', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            # Unauthorized user detected
            cv2.putText(frame, 'Unauthorized User', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Authentication', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
