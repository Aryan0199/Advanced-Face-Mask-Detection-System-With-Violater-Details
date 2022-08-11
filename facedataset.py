import cv2
import os

cam = cv2.VideoCapture(0) #opens default camera for capturing video
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #loading pre-trained haar cascade model to detect face

# For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

#start detect your face and take 30 pictures
while(True):

    ret, img = cam.read() #returns a tuple(return value, image); return val: boolean val to indicate whether reading was succesful
                          # image->img which is read by opencv
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting img into grayscale for easy processing
    faces = face_detector.detectMultiScale(gray, 1.3, 5) #detecting faces in terms of rectangle: (x,y,width,height)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  #draws a rectangle with red line of thikness 2px 
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video within 100ms to stop capturing
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break


print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


