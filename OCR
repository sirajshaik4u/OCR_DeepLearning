#____________________________Load data and pre-process_________________________
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import tensorflow as tf
tf.python.control_flow_ops = tf

#SET SEED:
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#LOAD DATA:
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#RESHAPE :[samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

#NORMALIZE: 0-255 => 0-1
X_train = X_train / 255
X_test = X_test / 255

#ONE-HOT ENCODE OUTPUTS:
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#Count of unique Chars:
num_classes = y_test.shape[1]

#_______________________________________Define and build model__________________________________
#DEFINE STRUCTURE OF NEURAL-NETWORK IN FN():
def larger_model():
	#CREATE MODEL:
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	#COMPILE MODEL
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


#BUILD THE MODEL:
model = larger_model()

#FIT THE MODEL:
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=80, batch_size=200, verbose=2)

#FINAL EVALUATION:
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#__________________________________Code to Save the Model________________________________________
from keras.models import model_from_json
import os

# serialize model to JSON
ocr_model_json = model.to_json()
with open("ocr_model_Final.json", "w") as json_file:
    json_file.write(ocr_model_json)
# serialize weights to HDF5
model.save_weights("ocr_model_Final.h5")
print("Saved model to disk")

#_________________________________Code to Load the model________________________________________
#IMPORT LIBS:
#from tensorflow.python.keras.models import model_from_json
from keras.models import model_from_json
import os

#LOAD JSON AND CREATE MODEL 
json_file = open('ocr_model_Final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#LOAD WEIGHTS INTO NEW MODEL
loaded_model.load_weights("ocr_model_Final.h5")
print("Loaded model from disk")

#EVALUATE LOADED MODEL ON TEST DATA 
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-score[1]*100))

#____________________________________Test the model with sample image___________________________________
#IMPORT LIBS:
import cv2
import numpy as np
from PIL import Image
import subprocess

#Draw image
fileName = "Test.jpg"
p = subprocess.Popen(["mspaint.exe", fileName])
returncode = p.wait() # wait for paint to exit

#LOAD-RESIZE-LOAD:
img1 = Image.open('Test.jpg')
img1.resize((28,28)).save("T28.jpg")
img = cv2.imread('T28.jpg')

#GRAY:
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#THRESHOLD:
#ret,img = cv2.threshold(img,96,255,cv2.THRESH_BINARY)
ret,img = cv2.threshold(img,96,255,cv2.THRESH_BINARY_INV)

#RESHAPE/ INT32 -> DEC:
img = img.reshape(28, 28).astype('float32')

#NORMALIZE:
img = img/255

#SPECIAL 'DS' AS/PER N/W:
img = np.array([[img]])

prediction = loaded_model.predict(img).argmax()
prediction
