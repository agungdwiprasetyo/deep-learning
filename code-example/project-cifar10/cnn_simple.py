# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# Accuracy: 71.82%

# ____________________________________________________________________________________________________
# Layer (type)                       Output Shape        Param #     Connected to
# ====================================================================================================
# convolution2d_1 (Convolution2D)    (None, 32, 32, 32)  896         convolution2d_input_1[0][0]
# ____________________________________________________________________________________________________
# dropout_1 (Dropout)                (None, 32, 32, 32)  0           convolution2d_1[0][0]
# ____________________________________________________________________________________________________
# convolution2d_2 (Convolution2D)    (None, 32, 32, 32)  9248        dropout_1[0][0]
# ____________________________________________________________________________________________________
# maxpooling2d_1 (MaxPooling2D)      (None, 32, 16, 16)  0           convolution2d_2[0][0]
# ____________________________________________________________________________________________________
# flatten_1 (Flatten)                (None, 8192)        0           maxpooling2d_1[0][0]
# ____________________________________________________________________________________________________
# dense_1 (Dense)                    (None, 512)         4194816     flatten_1[0][0]
# ____________________________________________________________________________________________________
# dropout_2 (Dropout)                (None, 512)         0           dense_1[0][0]
# ____________________________________________________________________________________________________
# dense_2 (Dense)                    (None, 10)          5130        dropout_2[0][0]
# ====================================================================================================
# Total params: 4210090
# ____________________________________________________________________________________________________


# 50000/50000 [==============================] - 24s - loss: 0.2515 - acc: 0.9116 - val_loss: 1.0101 - val_acc: 0.7131
# Epoch 21/25
# 50000/50000 [==============================] - 24s - loss: 0.2345 - acc: 0.9203 - val_loss: 1.0214 - val_acc: 0.7194
# Epoch 22/25
# 50000/50000 [==============================] - 24s - loss: 0.2215 - acc: 0.9234 - val_loss: 1.0112 - val_acc: 0.7173
# Epoch 23/25
# 50000/50000 [==============================] - 24s - loss: 0.2107 - acc: 0.9269 - val_loss: 1.0261 - val_acc: 0.7151
# Epoch 24/25
# 50000/50000 [==============================] - 24s - loss: 0.1986 - acc: 0.9322 - val_loss: 1.0462 - val_acc: 0.7170
# Epoch 25/25
# 50000/50000 [==============================] - 24s - loss: 0.1899 - acc: 0.9354 - val_loss: 1.0492 - val_acc: 0.7182