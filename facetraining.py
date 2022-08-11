import cv2
import numpy as np
from PIL import Image #pillow package
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create() # loading LBPH: Local Binary Patterns Histograms Model
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml"); #loading haarcascade_frontalface classifier

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #absolute img paths    
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8') #converting img to [Height,Width,Channel] format

        id = int(os.path.split(imagePath)[-1].split(".")[1])  #abs img path to User.1.1.jpg to [User,1,1] to 1
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids)) #training LBPH Model

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # saving trained model as traniner.yml

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids)))) 
