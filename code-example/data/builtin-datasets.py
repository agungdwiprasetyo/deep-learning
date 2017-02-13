# built-in datasets stored in ~/.keras/datasets/

# CIFAR10 small image classification
# Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# CIFAR100 small image classification
# Dataset of 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images
from keras.datasets import cifar100
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')


# IMDB Movie reviews sentiment classification
# Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative)
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.pkl", nb_words=None, skip_top=0, maxlen=None, test_split=0.1)


# Reuters newswire topics classification
# Dataset of 11,228 newswires from Reuters, labeled over 46 topics
from keras.datasets import reuters
(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.pkl", nb_words=None, skip_top=0, maxlen=None, test_split=0.1)


# MNIST database of handwritten digits
# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
