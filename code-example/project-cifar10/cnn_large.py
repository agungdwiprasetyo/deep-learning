# Large CNN model for CIFAR-10
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
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# Accuracy: 80.18%

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
# convolution2d_3 (Convolution2D)    (None, 64, 16, 16)  18496       maxpooling2d_1[0][0]
# ____________________________________________________________________________________________________
# dropout_2 (Dropout)                (None, 64, 16, 16)  0           convolution2d_3[0][0]
# ____________________________________________________________________________________________________
# convolution2d_4 (Convolution2D)    (None, 64, 16, 16)  36928       dropout_2[0][0]
# ____________________________________________________________________________________________________
# maxpooling2d_2 (MaxPooling2D)      (None, 64, 8, 8)    0           convolution2d_4[0][0]
# ____________________________________________________________________________________________________
# convolution2d_5 (Convolution2D)    (None, 128, 8, 8)   73856       maxpooling2d_2[0][0]
# ____________________________________________________________________________________________________
# dropout_3 (Dropout)                (None, 128, 8, 8)   0           convolution2d_5[0][0]
# ____________________________________________________________________________________________________
# convolution2d_6 (Convolution2D)    (None, 128, 8, 8)   147584      dropout_3[0][0]
# ____________________________________________________________________________________________________
# maxpooling2d_3 (MaxPooling2D)      (None, 128, 4, 4)   0           convolution2d_6[0][0]
# ____________________________________________________________________________________________________
# flatten_1 (Flatten)                (None, 2048)        0           maxpooling2d_3[0][0]
# ____________________________________________________________________________________________________
# dropout_4 (Dropout)                (None, 2048)        0           flatten_1[0][0]
# ____________________________________________________________________________________________________
# dense_1 (Dense)                    (None, 1024)        2098176     dropout_4[0][0]
# ____________________________________________________________________________________________________
# dropout_5 (Dropout)                (None, 1024)        0           dense_1[0][0]
# ____________________________________________________________________________________________________
# dense_2 (Dense)                    (None, 512)         524800      dropout_5[0][0]
# ____________________________________________________________________________________________________
# dropout_6 (Dropout)                (None, 512)         0           dense_2[0][0]
# ____________________________________________________________________________________________________
# dense_3 (Dense)                    (None, 10)          5130        dropout_6[0][0]
# ====================================================================================================
# Total params: 2915114
# ____________________________________________________________________________________________________


# 50000/50000 [==============================] - 34s - loss: 0.4993 - acc: 0.8230 - val_loss: 0.5994 - val_acc: 0.7932
# Epoch 20/25
# 50000/50000 [==============================] - 34s - loss: 0.4877 - acc: 0.8271 - val_loss: 0.5986 - val_acc: 0.7932
# Epoch 21/25
# 50000/50000 [==============================] - 34s - loss: 0.4714 - acc: 0.8327 - val_loss: 0.5916 - val_acc: 0.7959
# Epoch 22/25
# 50000/50000 [==============================] - 34s - loss: 0.4603 - acc: 0.8376 - val_loss: 0.5954 - val_acc: 0.8003
# Epoch 23/25
# 50000/50000 [==============================] - 34s - loss: 0.4454 - acc: 0.8410 - val_loss: 0.5742 - val_acc: 0.8024
# Epoch 24/25
# 50000/50000 [==============================] - 34s - loss: 0.4332 - acc: 0.8468 - val_loss: 0.5829 - val_acc: 0.8027
# Epoch 25/25
# 50000/50000 [==============================] - 34s - loss: 0.4217 - acc: 0.8498 - val_loss: 0.5785 - val_acc: 0.8018