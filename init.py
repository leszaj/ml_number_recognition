'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
# From https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
from __future__ import print_function
# the above line makes print functions compatible with Python 2
# so the code runs on both Python 2 and 3, you should use Python 3


from tensorflow import keras    # import keras

# Keras has some benchmark datasets ready built-in such as MNIST
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# We set the batch size and epoch hyper-parameters here
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Can you recognise what this is doing to the data?
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# It is always a good idea to print as you go along
# to get feel of the data, processing etc you are doing

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# this is your one-hot encoding, it takes indices of targets
# and converts them to vectors of size num_classes each
# where they have a 1 at the specified index
# this target is the digits, 0,1,2,3,4,5,6,7,8,9

# This is the network or model, our feed-forward network
# or multi-layer perceptron, can you tell how many layers it has?
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# notice how dropout is added after each hidden layer
# not a trick question but why isn't there dropout after the final layer?

model.summary() # this prints a nice summary of the model
# it also tells how many weights / parameters your model has

# We now select our loss function, the optimiser, and any extra metrics we want
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
# They use RMSProp and I left it as is in case you want to explore beyond the course
# but recognise it is just an extension of our vanilla gradient descent.

# This is the training loop, where we update the weights of the network
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
# Finally we get the loss and accuracy on our test set to see how well
# our model generalised or over-fitted etc.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])