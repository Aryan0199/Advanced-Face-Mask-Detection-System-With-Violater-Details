
import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime
import numpy as np
import os 
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode 

recognizer = cv2.face.LBPHFaceRecognizer_create() #load LBPH (Local Binary Pattern Histogram) model
recognizer.read('trainer/trainer.yml')   #load trained model
cascadePath = "haarcascade_frontalface_default.xml" 
faceCascade = cv2.CascadeClassifier(cascadePath) #load haar cascade classifier


font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter, the number of persons you want to include
id = 0 #two persons 


#names = ['','Aryan','Arush']  #key in names, start from the second place, leave first empty

# FETCHING EMPLOYEES DETAILS FROM DATABASE:
names=[]
try:
    connection=mysql.connector.connect(host='localhost',database='newdb',user='root',password='admin')

    sql_select_Query = "select * from employees"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    # get all records
    records = cursor.fetchall()
    for row in records:
        names.append(row[1]) #appending names of employees to array
        id+=1
       

except mysql.connector.Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if connection.is_connected():
        connection.close()
        cursor.close()
        print("MySQL connection is closed")

names.insert(0,'')

# # BUILDING CNN MODEL FROM SCRATCH TO CLASSIFY BETWEEN MASK AND NO MASK

# model=Sequential() #initiate sequential class to add variuos layers in sequence

# #FEATURE EXTRACTION
# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
# model.add(MaxPooling2D() ) #reduce the dimension of generated feautre map
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPooling2D() )
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPooling2D() )
# model.add(Flatten()) # converting feature map to 1d array

# #CLASSIFICATION
# model.add(Dense(100,activation='relu')) 
# model.add(Dense(1,activation='sigmoid'))

# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# #ImageDataGenerator lets to augment our images in real-time while model is training.
# from keras.preprocessing.image import ImageDataGenerator  
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory(
#         'train',
#         target_size=(150,150),
#         batch_size=16 ,
#         class_mode='binary')

# test_set = test_datagen.flow_from_directory(
#         'test',
#         target_size=(150,150),
#         batch_size=16,
#         class_mode='binary')

# model_saved=model.fit_generator(
#         training_set,
#         epochs=10,
#         validation_data=test_set,

#         )

# model.save('mymodel4.h5',model_saved)


#To test for individual images

mymodel=load_model('mymodel4.h5')
test_image=image.load_img(r'C:/Users/itsme/Documents/FaceMaskDetector/test/with_mask/1-with-mask.jpg',
                          target_size=(150,150,3))
test_image
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
mymodel.predict(test_image)[0][0]


# IMPLEMENTING LIVE DETECTION OF FACE MASK

mymodel=load_model('mymodel3.h5') #loading the trained CNN model for mask detection

cap=cv2.VideoCapture(0) #opens default camera for capturing video
# Define min window size to be recognized as a face
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _,img=cap.read()
    face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4) #detecting faces in terms of rectangle: (x,y,width,height)
                                                                           #minNeighbors: to reduce number of false positives
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg',face_img)
        test_image=image.load_img('temp.jpg',target_size=(150,150,3))
        test_image=image.img_to_array(test_image)#converts PIL (Python Image Library) img instance to a Numpy Array
        test_image=np.expand_dims(test_image,axis=0)
        pred=mymodel.predict(test_image)[0][0]  
        if pred==1:  #if person is NOT wearing MASK
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            # cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            ret, img =cap.read() 

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale( 
                    gray,
                    scaleFactor = 1.2,
                    minNeighbors = 5,
                    minSize = (int(minW), int(minH)),
                )
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

            for(x,y,w,h) in faces:

                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                    id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                    pid=id
                    # print('#######################  '+ pid)

                    # Check if confidence is less them 100 ==> "0" is perfect match 
                    if (confidence < 100):
                        id = names[id]
                        uname=str(id)
                        print("$$$$$$$$$$$$ "+ str(id))
                        print("$$$$$$$$$$$$ "+ str(len(names)))
                        #print("############################# "+ uname)
                        confidence = "  {0}%".format(round(100 - confidence))
                        
                    else:
                        id = "unknown"
                        confidence = "  {0}%".format(round(100 - confidence))
                    
                    cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2) #write username on image
                    cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  #wrtie the confidence level on image
                    if(id!='unknown'):
                        # STORING VIOLATERS DETAILS TO DATABASE
                        try:
                            connection=mysql.connector.connect(host='localhost',database='newdb',user='root',password='admin') 
                            
                            mySql_insert_query = "INSERT INTO defaulters (ID,Name,Date,Time) VALUES (%s,%s,%s,%s) "
                            val=(pid,uname,str(datetime.datetime.now().date()),str(datetime.datetime.now().time()))
                            cursor = connection.cursor()
                            cursor.execute(mySql_insert_query,val) #executing sql query
                            connection.commit()
                            print(cursor.rowcount, "Record inserted successfully into defaulters table")
                            cursor.close()
                        # except MySQLdb.IntegrityError:

                        except :
                               print("Failed to insert record into defaulters table {}")

                        finally:

                            if (connection.is_connected()):

                                connection.close()
                                print("MySQL connection is closed")
                                break
                        


                
            cv2.imshow('img',img) #show the image

                # k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
                # if k == 27:
                #     break


                
        else: #if person is wearing MASK
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
          
    cv2.imshow('img',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
