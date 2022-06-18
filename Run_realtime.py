import string
from cv2 import CAP_MSMF
from keras.models import load_model
from time import sleep
from keras.utils import img_to_array
from keras.preprocessing import image
from imutils.video import VideoStream
import cv2
import numpy as np
import time

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('FaceShape_Gender_Age_30epochs.h5')

# map labels for gender
gender_dict = {0:'Male', 1:'Female'}

# map labels for shape
shape_dict = {0:'Diamond', 1:'Round',2:'Oval',3:'Rectangle',4:'Heart'}

#source = 'sasa.mp4'
#cam = VideoStream(source).start()

cam = VideoStream(0,framerate=40).start()

while True:
    frame= cam.read()
    frame = cv2.resize(frame,[800,600])
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    labels=[]
 
    faces = face_classifier.detectMultiScale(grayFrame,1.3,2)

    for (x,y,w,h) in faces:
        roi_color=grayFrame[y-15:y+h+15,x-15:x+w+15]
        
        roi_color=cv2.resize(roi_color,(150,150),interpolation=cv2.INTER_AREA)
        
        gender_predict = model.predict(np.array(roi_color).reshape(1,150,150,1))
        #Get image ready for prediction
        roi=img_to_array(roi_color)
        roi=roi.astype('float32')/255  #Scale
        roi=np.expand_dims(roi,axis=0)  #Expand dims to get it ready for prediction (1, 48, 48, 1)

        pred = model.predict(roi)
        pred_gender = gender_dict[round(pred[0][0][0])]
        pred_age = round(pred[1][0][0])
        pred_shape = shape_dict[round(pred[0][0][0])]

        #print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age, "Predicted Shape:", pred_shape )
        
        resrult = "G: " + str(pred_gender) + " A: " + str(pred_age) + " S: " + str(pred_shape)
        position=(x,y-40)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(grayFrame,resrult,position,font,0.5,(255,255,255),1)

        cv2.rectangle(grayFrame,(x-15,y-15),(x+w+15,y+h+15),(50,50,255),2)

    cv2.imshow('showface', grayFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()