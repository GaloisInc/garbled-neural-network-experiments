import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import SGD, RMSprop, Adam
from keras import layers

def custom_activation(x):
    return x*x

get_custom_objects().update({'custom_activation': Activation(custom_activation)})



def SecureML_MLP_MNIST(alternate=False):
    # Note, square only used in prediction--not training (p 21)
    model = Sequential()
    model.add(Dense(128,input_shape=(28*28,)))
    model.add(Activation('relu')) #Activation(custom_activation)
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(128))
    if alternate:
        model.add(Activation('tanh')) #Activation(custom_activation)
    else:
        model.add(Activation('relu')) #Activation(custom_activation)
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))


    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

    return model


def CryptoNets_CNN_MNIST(pooling="max",alternate=False):
    # Description of model in paper is nonsensical; so I altered it.
    #
    # Note: average pooling used in paper
    # CryptoNets guys super optimize their model after training;
    # combined layers and such.
    # Also, they pad images asymmetrically: one row left, one row up.

    # TODO: Check if new model pads
    pool = MaxPooling2D if pooling == "max" else AveragePooling2D

    model = Sequential()
    model.add(Conv2D(5, kernel_size=(5, 5),strides=(1,1),
                     input_shape=(28,28,1)))
    model.add(Activation('relu'))#custom_activation))
    model.add(pool(pool_size=(3, 3)))
    model.add(Conv2D(10, (3, 3)))
    if alternate:
        model.add(Activation('tanh'))#custom_activation))
    else:
        model.add(Activation('relu'))#custom_activation))
    model.add(pool(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    if alternate:
        model.add(Activation('tanh'))#custom_activation))
    else:
        model.add(Activation('relu'))#custom_activation))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    return model

def MiniONN_CNN_MNIST(alternate=False):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    if alternate:
        model.add(Dense(100, activation='tanh'))
    else:
        model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def MiniONN_CNN_CIFAR(pooling="max",alternate=False):
    pool = MaxPooling2D if pooling == "max" else AveragePooling2D
    activation = 'tanh' if alternate else 'relu'

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(32,32,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(pool(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(pool(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation=activation))
    model.add(Conv2D(64, (1, 1), activation=activation))
    model.add(Conv2D(16, (1, 1), activation=activation))
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    opt = keras.optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def DeepSecure_CNN_MNIST(alternate=False):
    # I assume asymmetric padding was used during experiments.
    # In this case, I adjust kernel_size to remove need for padding.
    model = Sequential()
    model.add(Conv2D(5, kernel_size=(4, 4),strides=(2,2),
                     activation='relu',
                     input_shape=(28,28,1)))
    model.add(Flatten())
    if alternate:
        model.add(Dense(100, activation='tanh'))
    else:
        model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def test_MLP_MNIST():
    model = Sequential()
    model.add(Dense(10, input_shape = (28*28,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

    return model
