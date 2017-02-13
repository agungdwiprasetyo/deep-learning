# Deep Learning
Deep learning dalam repository ini menggunakan framework Keras yang berjalan pada TensorFlow. Deep Learning digunakan untuk menggali data yang berukuran sangat besar yang menggunakan teknik-teknik machine learning, terutama neural network.

## Requirements
Menggunakan bahasa pemrograman Python, baik versi 2.7 maupun versi 3.5. Lalu sangat disarankan menggunakan sistem operasi Linux 64-bit.

#### Proses instalasi TensorFlow pada python:

- Install pip
```sh
$ sudo apt-get install python-pip python-dev
```
- Install TensorFlow
```sh
$ sudo pip install tensorflow 		# apabila hanya menggunakan cpu untuk pemrosesannya
$ sudo pip install tensorflow-gpu	# apabila support penggunaan GPU seperti CUDA pada NVIDIA
```
- Download Binary file
```sh
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0rc2-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0rc2-cp27-none-linux_x86_64.whl

# 64-bit, CPU only, Python 3.5
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0rc2-cp35-cp35m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0rc2-cp35-cp35m-linux_x86_64.whl
```
- Install Binary file yang telah didownload sebelumnya dengan pip
```sh
# Python 2
$ sudo pip install --upgrade $TF_BINARY_URL

# Python 3
$ sudo pip3 install --upgrade $TF_BINARY_URL
```
- Tes hasil instalasi
```sh
$ python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```

#### Proses instalasi Keras
Install Keras menggunakan Pip
```sh
$ sudo pip install keras
```
Untuk melihat versi Keras yang terinstal
```sh
$ python -c "import keras; print keras.__version__"
```

## Mulai
Struktur data yang digunakan dalam Keras yaitu berupa model. Model yang sering digunakan yaitu Sequential.
```python
#!/usr/bin/env python
from keras.models import Sequential
model = Sequential()
```