import cv2
import os
images=[]
targets=[]
prefix_dir_angry= r'C:/Users/mukhe/Facedetection/images/train/angry'
prefix_dir_disgust=r'C:/Users/mukhe/Facedetection/images/train/disgust'
prefix_dir_fear=r'C:/Users/mukhe/Facedetection/images/train/fear'
prefix_dir_happy=r'C:/Users/mukhe/Facedetection/images/train/happy'
prefix_dir_neutral=r'C:/Users/mukhe/Facedetection/images/train/neutral'
prefix_dir_sad=r'C:/Users/mukhe/Facedetection/images/train/sad'
prefix_dir_surprise=r'C:/Users/mukhe/Facedetection/images/train/surprise'
content=os.listdir(prefix_dir_angry)

for image in content:
    try:
        image_path=prefix_dir_angry + '/'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(100,100))
        images.append(resized_image)
        targets.append("angry")
    except Exception as e:
        print("exception", e)


content=os.listdir(prefix_dir_disgust)

for image in content:
    try:
        image_path=prefix_dir_disgust + '/'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(100,100))
        images.append(resized_image)
        targets.append("disgust")
    except Exception as e:
        print("exception", e)
content=os.listdir(prefix_dir_fear)

for image in content:
    try:
        image_path=prefix_dir_fear + '/'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(100,100))
        images.append(resized_image)
        targets.append("fear")
    except Exception as e:
        print("exception", e)
content=os.listdir(prefix_dir_happy)

for image in content:
    try:
        image_path=prefix_dir_happy + '/'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(100,100))
        images.append(resized_image)
        targets.append("happy")
    except Exception as e:
        print("exception", e)
content=os.listdir(prefix_dir_neutral)

for image in content:
    try:
        image_path=prefix_dir_neutral + '/'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(100,100))
        images.append(resized_image)
        targets.append("neutral")
    except Exception as e:
        print("exception", e)
content=os.listdir(prefix_dir_sad)

for image in content:
    try:
        image_path=prefix_dir_sad + '/'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(100,100))
        images.append(resized_image)
        targets.append("sad")
    except Exception as e:
        print("exception", e)
content=os.listdir(prefix_dir_surprise)

for image in content:
    try:
        image_path=prefix_dir_surprise + '/'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(100,100))
        images.append(resized_image)
        targets.append("surprise")
    except Exception as e:
        print("exception", e)
import numpy as np
X=np.array(images)
Y=np.array(targets)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8)
X_train.ndim
X_train.shape
X_train.size
X_train=X_train.reshape(X_train.shape[0],100, 100, 1)
X_test=X_test.reshape(X_test.shape[0],100,100,1)
X_train=X_train/255
X_test=X_test/255
Y_train
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y_train2 = label_encoder.fit_transform(Y_train)
Y_test2=label_encoder.fit_transform(Y_test)
Y_train2
from keras.utils import np_utils

Y_train2=np_utils.to_categorical(Y_train2)
Y_train2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
model= Sequential()
model.add(Conv2D(200,(3,3), activation='relu'))
model.add(Conv2D(150,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(100,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout((0.25)))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cp= ModelCheckpoint('model-best', verbose=0,save_best_only=True)
model.fit(X_train,Y_train2, epochs=15 ,callbacks=[cp], validation_split=0.2)

import cv2
import numpy as np
from keras.models import load_model
model=load_model('model-best')
face_detect=cv2.CascadeClassifier(r'C:\Users\mukhe\anaconda3\Lib\site-packages\cv2\haarcascade_frontalface_default.xml')
source=cv2.VideoCapture(0)

while 1:
    _,image=source.read()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.5,5)
    for x,y,w,h in faces:
        face_roi=gray[y:y+w, x:x+w]
        resized_face=cv2.resize(face_roi,(100,100))
        normalized_face=resized_face/255
        reshaped_face=np.reshape(normalized_face,(1,100,100,1))
        result=model.predict(reshaped_face)[0]
        if np.amax(result)==result[0]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image,"Angry",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
        if np.amax(result)==result[1]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image,"digusted",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
        if np.amax(result)==result[2]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image,"fear",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
        if np.amax(result)==result[3]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image,"happy",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
        if np.amax(result)==result[4]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image,"neutral",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
        if np.amax(result)==result[5]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image,"sad",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
        if np.amax(result)==result[6]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image,"surprised",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
       
     
    cv2.imshow('Emotion detection',image)
    key=cv2.waitKey(1)
    if key==27:
        break
        
cv2.destroyAllWindows()
source.release()