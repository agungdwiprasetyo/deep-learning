# Example of one hot encoding
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy
import pandas
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("../data/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4]
Y = dataset[:,4]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# create model
model = Sequential()
model.add(Dense(12, input_dim=4, init='normal', activation='relu'))
model.add(Dense(3, init='normal', activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, dummy_y, validation_split=0.33, nb_epoch=100, batch_size=10)
