import tensorflow as tf
import os
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import cv2

model_file = "emotion.h5"
model = tf.keras.models.load_model(model_file)

# print(model.summary())

batchSize = 32 

# get categories of the model
print("categories: ")
trainPath = "data/train"
categories = os.listdir(trainPath)
categories.sort()
print(categories)
numOfClasses = len(categories)
print(numOfClasses)

# find the face inside the image
def findFace(pathOfImage):
    image = cv2.imread(pathOfImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    haarcascadeFile = "haarcascade_frontalface_default.xml"  # Corrected file extension
    face_cascade = cv2.CascadeClassifier(haarcascadeFile)
    
    # Check if the Haar cascade file was loaded successfully
    if face_cascade.empty():
        print(f"Error loading cascade file {haarcascadeFile}")
        return None
    
    faces = face_cascade.detectMultiScale(gray)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #we only want to capture the face
        onlyFace =gray[y:y+h, x:x+w]
        
    # Resize the original image (for example, make it 50% smaller)
    image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
    
    return onlyFace
    # Show the sized image with rectangles around detected faces
    # cv2.imshow("Image", gray)
    # cv2.waitKey(0)  # Keep the window open until a key is pressed
    
def PrepareModel(faceImage):
    #1. resize it to 48x48 as our model wants that
    resized = cv2.resize(faceImage, (48,48), interpolation=cv2.INTER_AREA)
    #expand the image to create batch of images
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.0
    return imgResult
    
# Test the image
testImagePath = "angry.png"
faceGrayImage = findFace(testImagePath)

imgForModel = PrepareModel(faceGrayImage)


#run teh prediction
resultArray = model.predict(imgForModel, verbose=1)
answers = np.argmax(resultArray, axis=1)

print(answers[0])

test = categories[answers[0]]

print("Prediction: " + test)

#show the image with the test 
img = cv2.imread(testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, test, (0, 20), font, 0.5, (209, 19, 77), 2)
# Show the sized image with rectangles around detected faces

# Resize the image to half its original size
# img = cv2.resize(img, (int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)))

cv2.imshow("Image", img)
cv2.waitKey(0)  # Keep the window open until a key is pressed
cv2.destroyAllWindows()