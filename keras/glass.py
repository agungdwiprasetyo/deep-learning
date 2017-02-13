#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)
# load glass training dataset
dataset = numpy.loadtxt("datasets/glasstraining.csv", delimiter=",")

X = dataset[:,0:4]
Y = dataset[:,4]

# create model
model = Sequential()
model.add(Dense(12, input_dim=4, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
# use sigmoid function
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x) for x in predictions]
print(rounded)
