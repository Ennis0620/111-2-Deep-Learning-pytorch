from keras import utils
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,ReLU
from keras.optimizers import Adam
import tensorflow as tf
import seaborn as sns

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train:',x_train.shape)
print('y_train:',y_train.shape)
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)

y_train = y_train.flatten()
y_test = y_test.flatten()
print('y_train flatten:',y_train.shape)
print('y_test flatten:',y_test.shape)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']


def construct_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), padding="same", activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="valid", activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


class net(Model):
    def __init__(self,in_classes):
        super(net,self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), padding="same", activation='relu')

    def call(self,layer):
        layer = self.conv1(layer)
        
        return layer

#model=construct_model()
model1 = net(in_classes=10)
model1.summary()

import visualkeras
from PIL import ImageFont
visualkeras.layered_view(model1, legend=True)


'''
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy(),
              metrics=['val_acc'])

model.fit(x_train,y_train,
          batch_size=512,
          epochs=1000,
          )
'''