# coding=UTF-8
import cv2
import pickle  #对象序列化：它是一个将任意复杂的对象转成对象的文本或二进制表示的过程
import os.path 
import numpy as np
from imutils import paths  #提供一系列便捷功能进行基本的图像处理功能
from sklearn.preprocessing import LabelBinarizer   #标签二值化LabelBinarizer 
from sklearn.model_selection import train_test_split #随机划分训练集和测试集
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit  #自己写的程序重按比例放大缩小图片


LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    # Resize the letter so it fits in a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)  #按比例裁剪

    # Add a third channel dimension to the image to make Keras happy为啥呢
    image = np.expand_dims(image, axis=2)  #在axis的那一个轴上把数据加上去

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2]  #定义标签

    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)  归一化之后放在数组中
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
print(lb)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)   #为什么是Y_train二值化之后的标签给他？
print(np.shape(Y_train))

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)    #pickle.dump( object, file)，pickle对象到一个打开的文件 

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))  #shurudaxaio
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(32, activation="softmax"))   #genjubiaoian

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
