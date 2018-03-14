import tensorflow as tf
import numpy as np
import keras as K

from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical

from keras.datasets import cifar10

(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()
Y_train_onehot = to_categorical(Y_train)
Y_test_onehot = to_categorical(Y_test)

model = Sequential()

model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train_onehot,epochs=10,batch_size=64)

score = model.evaluate(X_test,Y_test_onehot,batch_size=64)

print("\nAccuracy of TestSet=",score[-1])