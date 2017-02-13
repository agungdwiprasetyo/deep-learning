# Use scikit-learn to grid search learning rates

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import numpy
import pandas

# Function to create model, required for KerasClassifier
def create_model(lrate=0.01, momentum=0.0, decay=0.0):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, init='normal', activation='relu'))
	model.add(Dense(8, init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# optimizer
	sgd = SGD(lr=lrate, momentum=momentum, decay=decay)
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("../data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=5, verbose=0)
# grid search epochs, batch size and optimizer
lrate = [0, 0.2, 0.4, 0.6, 0.8]
momentum = [0, 0.1, 0.2, 0.3]
decay = [0, 0.1, 0.001]
param_grid = dict(lrate=lrate, momentum=momentum, decay=decay)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
