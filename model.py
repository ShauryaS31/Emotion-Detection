import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt



trainPath = "data/train"
testPath  = "data/test"

# Get list of folder names (classes) and sort them
folderlist = os.listdir(trainPath)
folderlist.sort()
# print(folderlist)

X_train = []
y_train = []

#load the training data in to an array       
for i, category in enumerate(folderlist):
    # print(f"Processing folder: {category} with label: {i}")
    files = os.listdir(trainPath + "/" + category)
    # print(f"Found {len(files)} files in {category}") 
    for file in files:
        img = cv2.imread(trainPath + "/" + category + "/{0}".format(file), 0)
        if img is not None:  # Check if image is loaded correctly
            X_train.append(img)
            y_train.append(i)  # Append the label corresponding to the folder
        else:
            print(f"Error reading image {file} in {category}")  # Debug if image fails to load
        
# print(len(X_train)) #28709 images


#show the first image using cv2
"""img1 = X_train[0]
cv2.imshow("img1", img1)
cv2.waitKey(0)"""

# check the labled
"""label_counts = Counter(y_train)
print("Labels distribution in training data:", label_counts)"""




#do the same thing for test data
X_test = []
y_test = []

folderlist = os.listdir(testPath)
folderlist.sort()



#load the test data in to an array       
for i, category in enumerate(folderlist):
    # print(f"Processing folder: {category} with label: {i}")
    files = os.listdir(testPath + "/" + category)
    # print(f"Found {len(files)} files in {category}") 
    for file in files:
        img = cv2.imread(testPath + "/" + category + "/{0}".format(file), 0)
        if img is not None:  # Check if image is loaded correctly
            X_test.append(img)
            y_test.append(i)  # Append the label corresponding to the folder
        else:
            print(f"Error reading image {file} in {category}")  # Debug if image fails to load

# print("TEST DATA:")        
# print(X_test) 
# print(len(X_test)) #7178

#conver the data into numpy arrays
X_train = np.array(X_train, "float32")
y_train = np.array(y_train, "float32")
X_test = np.array(X_test, "float32")
y_test = np.array(y_test, "float32")



#checking the shapes of x_train and the first image
# print(X_train.shape)
# print(X_train[0]) #48x48



# normilise the data- values of each images should be between 0 and 1
# another dimension to the data : (28709,48,48,1)

X_train = X_train / 255.0  #to normalise it to 1
X_test =  X_test/255.0 

#to reshape the images
numOfImages = X_train.shape[0] 
X_train = X_train.reshape(numOfImages, 48, 48, 1)#add another dim

# print(X_train[0])
# print(X_train.shape)


# same for teh test 
numOfImages = X_test.shape[0] 
X_test = X_test.reshape(numOfImages, 48, 48, 1)
# print(X_test.shape)


#convert the lables to categorical labels
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# print(y_train)
# print("------------------------")
# print(y_train.shape)
# print("------------------------")
# print(y_train[0])



# BUILD THE CNN MODEL 
input_shape = X_train.shape[1:]  
# print(input_shape) #(48, 48, 1)
## Building the CNN architecture
model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters= 48, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters= 48, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters= 128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters= 128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters= 256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters= 256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters= 256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters= 512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters= 512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(4056, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4056, activation="relu"))
model.add(Dense(7, activation="softmax"))


# print(model.summary())

# # Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#defining the batch size- # image trained in every batch
batch=32
epochs=30

# Calculate the # steps (batches) per epoch based on total training samples and batch size
stepsPerEpoch = np.ceil(len(X_train)/batch)
validationSteps = np.ceil(len(X_test)/batch)

#if the val_accuracy was not improved over 5 itrations then teh trainning will stop
stopEarly = EarlyStopping(monitor="val-accuracy", patience=5, mode='max')

#train the model
history = model.fit(X_train,
                    y_train,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    callbacks=[stopEarly])

#how the result based on pyplot
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

#Show hte chart
epochs=range(len(acc))

#show train and validation train chart
plt.plot(epochs, acc, "r", label="Train accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend(loc="lower right")
plt.show()

#show the loss and validation loss chart
plt.plot(epochs, loss, "r", label="Train loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend(loc="upper right")
plt.show()

#Save the model
modelFileName = "emotion.h5"
model.save(modelFileName)




