from keras.datasets import mnist
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
import scipy.misc
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
#sys.stdin.flush()
train,test = mnist.load_data()
train_x,train_y = train
train_x2 = train_x
# start
"""plt.subplot(221)
plt.imshow(train_x[0], cmap=plt.get_cmap('gray'))
plt.show()"""
# end
test_x,test_y = test
test_x2 = test_x
shape1 = train_x.shape
shape2 = test_x.shape
train_x = train_x.reshape(shape1[0],shape1[1]*shape1[2])
test_x = test_x.reshape(shape2[0],shape2[1]*shape2[2])
train_x = train_x / 255
test_x = test_x / 255
train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)
print(train_x[14].shape)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

model = Sequential( [
    Dense(784,input_shape=(784,)),
    Activation('relu'),
    Dense(50),
    Activation('relu'),
    Dense(20),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
] )
model.summary()
model.compile(Adam(lr=0.005),loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(train_x,train_y,epochs=10,batch_size=250)
score = model.evaluate(test_x,test_y)
print("Accuracy is : {}".format(score[1]*100) )
#model.fit()
print(train_x[14].shape)

for i in range(1,10):
    p=random.randint(1,1000)
    predictions=model.predict(np.array([test_x[p]]) )
    print(np.argmax(predictions))
    plt.imshow(test_x2[p], cmap=plt.get_cmap('gray'))
    plt.show()
