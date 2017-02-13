# CNN for MNIST hand written digit recognition
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][channels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# create model
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# define data preparation
datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, zca_whitening=True)
# fit parametres from data
datagen.fit(X_train)
# prepare the data generator for iteration
datagen.flow(X_train, y_train, batch_size=64)
# Fit the model
model.fit_generator(datagen, validation_data=(X_test, y_test), samples_per_epoch=len(X_train), nb_epoch=5)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Classification Error: %.2f%%" % (100-scores[1]*100))
