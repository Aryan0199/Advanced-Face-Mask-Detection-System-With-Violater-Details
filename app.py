
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
# import mysql.connection
from mysql.connector import Error
from mysql.connector import errorcode 
from flask import Flask,redirect, render_template,url_for
from flask_mysqldb import MySQL
import MySQLdb.cursors
# from flask_sqlalchemy import SQLAlchemy
import subprocess as sp
# global connection2
app= Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'admin'
app.config['MYSQL_DB'] = 'newdb'
mysql=MySQL(app)
# def create_server_connection(host_name, user_name, user_password,database_name):
#     connection = None
#     try:
#         connection = mysql.connector.connect(
#             host=host_name,
#             user=user_name,
#             passwd=user_password,
#             database=database_name
#         )
#         #print("MySQL Database connection successful")
#     except Error as err:
#         print(f"Error: '{err}'")

#     return connection
# connection = create_server_connection("127.0.0.1", "root", "admin", "newdb")
# @app.route("/records",methods=['GET','POST'])
# def userRecord():
#     cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#     cursor.execute("""SELECT * FROM defaulters ORDER BY Date,Time DESC""")
#     data=cursor.fetchall()
#     return render_template('records.html',data=data)

    
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/cnn")
def cnn():
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
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)                  
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
    return redirect(url_for("home"))

@app.route("/hybrid")
def hybrid():
    # FETCHING EMPLOYEES DETAILS FROM DATABASE:
    # cur = mysql.connection.cusrsor()
    # cur.execute("SELECT * from employes")
    names=[]
    recognizer = cv2.face.LBPHFaceRecognizer_create() #load LBPH (Local Binary Pattern Histogram) model
    recognizer.read('trainer/trainer.yml')   #load trained model
    cascadePath = "haarcascade_frontalface_default.xml" 
    faceCascade = cv2.CascadeClassifier(cascadePath) #load haar cascade classifier
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0 #two persons 
    
    
    # connection2=mysql.connector.connect(host='localhost',database='newdb',user='root',password='admin')

    sql_select_Query = "select * from employees"
    cursor = mysql.connection.cursor()
    cursor.execute(sql_select_Query)
    
    # get all records
    records = cursor.fetchall()
    cursor.close()
    print(records)
    for row in records:
        names.append(row[1]) #appending names of employees to array
        print("$$$$$$$$$$ FETCHED")
        id+=1
        

    names.insert(0,'')

    mymodel=load_model('mymodel3.h5')

    cap=cv2.VideoCapture(0)
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
                            print("$$$$$$$$$$$$ "+ str(id))
                            print("$$$$$$$$$$$$ "+ str(len(names)))
                            id = names[id]
                            uname=str(id)
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
                            
                                mySql_insert_query = "INSERT INTO defaulters (ID,Name,Date,Time) VALUES (%s,%s,%s,%s) "
                                val=(pid,uname,str(datetime.datetime.now().date()),str(datetime.datetime.now().time()))
                                cursor = mysql.connection.cursor()
                                cursor.execute(mySql_insert_query,val) #executing sql query
                                mysql.connection.commit()
                                print(cursor.rowcount, "Record inserted successfully into defaulters table")
                                cursor.close()
                            # except MySQLdb.IntegrityError:
                            except:
                                print("Defaulter details already added")



                        
                            


                    
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
    return redirect(url_for("home"))

@app.route('/records',methods=['GET','POST'])
def records():
    #creating variable for connection
    cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    #executing query
    sql_query='select * from defaulters where Name =%s ORDER BY Date DESC'

    #Executing Query
    cursor.execute(sql_query,('Aryan',))
    #fetching all records from database
    data=cursor.fetchall()
    #returning back to projectlist.html with all records from MySQL which are stored in variable data
    return render_template("records.html",data=data)

if __name__=="__main__":
    app.run()

