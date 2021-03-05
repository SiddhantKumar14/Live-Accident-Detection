from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv3D, MaxPooling3D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import GlobalMaxPool2D
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Reshape, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.utils import np_utils, generic_utils

import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
from sklearn import preprocessing

to_predict = []
classes = ['Crash', 'NotCrash']

model.load_weights('model-sgd-0001-20epochs.h5')
num_frames = 0
cap = cv2.VideoCapture('/home/siddhant/Datasets/Crash Dataset/crashCompilationTesting.mp4')

#cap.set(12, 50)
#cap.set(6, 10)
#cap.set(cv2.CAP_PROP_FPS, 10)

preds = []

classe = ''
import time
fps = 30
counter  = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_cp = frame
    
    frame_cp = cv2.resize(frame, (144, 96))
    
    if counter%5 == 0:
        to_predict.append(frame_cp)
    counter += 1
    predict = 0
    if len(to_predict) == 30:

        frame_to_predict = [[]]
        frame_to_predict[0] = np.array(to_predict, dtype=np.float32)


        predict = model.predict(np.array(frame_to_predict))
        classe = classes[np.argmax(predict)]
        if np.amax(predict) > 0.60:
            print('Class = ',classe, 'Precision = ', np.amax(predict)*100,'%')
            preds.append(np.argmax(predict))
           # with open('gesture.pkl','wb') as f:
                #pickle.dump(np.argmax(predict), f)
        if len(preds) >= 10:
            preds = preds[8:9]

        to_predict = []
        time.sleep(0.1)
        font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),1,cv2.LINE_AA)
    
    cv2.imshow('Live Accident Detection',frame)
    if cv2.waitKey(int( (1 / int(fps)) * 1000)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()