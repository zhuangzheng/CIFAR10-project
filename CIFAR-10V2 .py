'''
这是利用卷积神经网络实现的第二个版本，包含两个卷积块
'''

import tensorflow as tf
import numpy as np
import keras as K

from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical

from keras.datasets import cifar10

# # # 载入数据
(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()
# # # 将Y转换成one_hot向量
Y_train_onehot = to_categorical(Y_train)
Y_test_onehot = to_categorical(Y_test)

model = Sequential()

# # # 第一个卷积块包含32个过滤器，大小为5X5，步长为1，输出不改变尺寸，再使用2X2的最大池化层将输出长宽各缩小一半，再连接一个规范层
model.add(Conv2D(filters=32,kernel_size=5,padding='same',input_shape = (32,32,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(BatchNormalization())

# # # 第二个卷积块和第一个结构类似，由于尺寸缩小，过滤器的数量增加了一倍
model.add(Conv2D(filters=64,kernel_size=5,padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(BatchNormalization())

# # # 将图片展开之后连接一个长度为128的全连接层
model.add(Flatten())
model.add(Dense(128))

# # # 这里使用了一个Dropout层随机关闭神经元，降低过拟合概率
model.add(Dropout(rate=0.75))

# # # 输出层使用长度为10的全连接层和softmax函数激活
model.add(Dense(10))
model.add(Activation('softmax'))

# # # 优化算法使用adam，加速梯度下降过程
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# # # 训练时使用100个epochs,batch大小设置为32个，用于降低计算量，因为我的显卡版本比较老
model.fit(X_train,Y_train_onehot,epochs=100,batch_size=32)

# # # 在测试集上评估模型效果
score = model.evaluate(X_test,Y_test_onehot,batch_size=32)

print("\nAccuracy of TestSet=",score[-1])
