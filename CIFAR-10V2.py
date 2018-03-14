'''
这是一个对CIFAR-10数据集进行分类的算法的模型训练程序，
在网络结构上比较简单的采取了两个FC块，作为与之后其他
算法效果的比较基准
 '''


import tensorflow as tf
import numpy as np
import keras as K

from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical

from keras.datasets import cifar10

# # # 载入CIFAR-10数据
(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()

# # # 将标签转化为one_hot向量
Y_train_onehot = to_categorical(Y_train)
Y_test_onehot = to_categorical(Y_test)

model = Sequential()

# # # 第一个块包含一个展开层，一个FC层，一个激活层和一个规范层，激活层使用relu函数，和规范层一样，目的都是为了避免梯度消失
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

# # # 第二个块和上一层结构类似，使用两个隐含层可以表达更复杂的函数关系
model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())

# # # 输出层使用softmax函数作为激活函数
model.add(Dense(10))
model.add(Activation('softmax'))

# # # 优化算法使用了Adam，Adam算法相比梯度下降算法能明显提高优化速度
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train_onehot,epochs=10,batch_size=64)

score = model.evaluate(X_test,Y_test_onehot,batch_size=64)

print("\nAccuracy of TestSet=",score[-1])