# CIFAR10-project
数据集来自于CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10是多伦多大学的一个彩色图片数据集，像素尺寸为32x32,包含60000个样本，10个标签，每一个标签下包含6000个样本，内容为汽车，飞机及各种动物，

其中训练集包含50000个，测试集10000，两者分布完全相同，输入维度为（m,3,32,32）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
v1版本网络结构:
    Conv1(32个filter,大小为5x5）
    Relu1
    BN1
    Maxpooling1
    
    Conv2(64个filter,大小为5x5）
    Relu2
    BN2
    Maxpooling2
    
    FC1(128)
    FC2(10)
    Softmax

损失函数为catagorical_crossentropy,优化器为adam
epoch=10,batchsize=64
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
960跑这个真是苦不堪言...
