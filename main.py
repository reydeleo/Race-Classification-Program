import keras
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input
from pyglet.compat import izip_longest
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

numSamples = 20
numEpochsInEachFold = 3
batchSize = 10


def grouper(iterable, n, fillvalue=None):
    # "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

#############################################################################

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-1]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(0.0001), metrics=['acc'])
# Show a summary of the model. Check the number of trainable parameters
model.summary()

#############################################################################

train_path = 'white-and-mexican/train'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['mexican', 'white'], batch_size=numSamples)
batchList = next(train_batches)
images = list()
labels = list()
tuples = list()

# making tuple list
for i in range(numSamples):
    t = (batchList[0][i], batchList[1][i])
    tuples.append(t)

trainImages = []
trainLabels = []
testImages = []
testLabels = []

groupSize = numSamples/10
groupSize = int(groupSize)
split_data = list(grouper(tuples, groupSize, 0))
random.shuffle(tuples)

# array to store accuracies of 10 fold
accuracies = []
histories = []
for i in range(10):
    # this is going to give the test group
    group = split_data[i]
    # preparing the tuples so that they can be fed into the network
    # inserting images into testImages and labels into testLabels
    numInEachGroup = int(numSamples/10)
    for j in range(numInEachGroup):
        testImages.append(group[j][0])
        testLabels.append(group[j][1])
    # making the lists into numpy arrays
    testImages = np.array(testImages)
    testLabels = np.array(testLabels)
    # preprocessing testImages
    for a in range(numInEachGroup):
        testImages[a] = np.expand_dims(testImages[a], axis=0)
        testImages[a] = preprocess_input(testImages[a])
    # a list to store the training data
    restOfData = []
    # putting together the training set of all of the other data
    for foldIndex in range(10):
        if foldIndex != i:
            for element in range(numInEachGroup):
                restOfData.append(split_data[foldIndex][element])
    # taking apart the tuples in restOfData and putting in separate lists for training
    trainingDataSize = numSamples-numInEachGroup
    for b in range(trainingDataSize):
        trainImages.append(restOfData[b][0])
        trainLabels.append(restOfData[b][1])
    # making the lists into numpy arrays
    trainImages = np.array(trainImages)
    trainLabels = np.array(trainLabels)
    # preprocessing trainImages
    for imgIndex in range(trainingDataSize):
        trainImages[imgIndex] = np.expand_dims(trainImages[imgIndex], axis=0)
        trainImages[imgIndex] = preprocess_input(trainImages[imgIndex])
    # training model
    hstory = model.fit(x=trainImages, y=trainLabels, batch_size=batchSize, epochs=numEpochsInEachFold)
    histories.append(hstory)
    # predictions of model
    predictions = model.predict(x=testImages)
    print("Predictions:")
    print()
    print(predictions)
    print()
    print("real labels:")
    tl = np.argmax(testLabels, axis=1)
    print(tl)
    print()
    print("predicted labels:")
    pl = np.argmax(predictions, axis=1)
    print(pl)
    print()
    print("Accuracy:")
    accuracy = sum(tl == pl) / len(pl)
    accuracies.append(accuracy)
    print(accuracy)
    print("###############")
    print()
    # resetting the lists
    trainImages = []
    trainLabels = []
    testImages = []
    testLabels = []


# final accuracy
accuracies = np.array(accuracies)
finalAccuracy = np.mean(accuracies)
print("Final Accuracy: ")
print(finalAccuracy)


acc = hstory.history['acc']
loss = hstory.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()


plt.show()