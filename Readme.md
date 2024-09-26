## EMOTION DETECTION 

Dataset- https://www.kaggle.com/datasets/msambare/fer2013?select=train

## STEPS:
    1. Import the libraries
        - numpy, tensorflow, cv2, os, etc.

    2. Setup the train and test folders
        - Define paths for training and testing datasets.

    3. Load the train data into an array
        - Iterate through folders, read images, and store them in X_train and y_train.

    4. Load the test data into an array
        - Similar process as for training data, storing in X_test and y_test.

    5. Convert the data into numpy arrays
        - Convert lists to numpy arrays for easier manipulation.

    6. Check the shapes of X_train
        - Verify the dimensions of the training dataset.

    7. Normalize the data
        - Scale pixel values to be between 0 and 1.

    8. Add another dimension to the data
        - Reshape X_train to be of shape (28709, 48, 48, 1).

    9. Do the same for test data
        - Reshape X_test similarly.

    10.Convert the labels to categorical labels
        - Use one-hot encoding for y_train and y_test.


## Build the CNN Model 
- Input Layer: Shape (48, 48, 1).

#### Building the CNN architecture
- Convolutional Layers: 
    Layer 1: 48 filters, (3, 3), ReLU.
    Layer 2: 48 filters, (3, 3), ReLU.
    Pooling Layer 1: MaxPooling (2, 2), strides (2, 2).
    Layer 3: 128 filters, (3, 3), ReLU.
    Layer 4: 128 filters, (3, 3), ReLU.
    Pooling Layer 2: MaxPooling (2, 2), strides (2, 2).
    Layer 5: 256 filters, (3, 3), ReLU.
    Layer 6: 256 filters, (3, 3), ReLU.
    Layer 7: 256 filters, (3, 3), ReLU.
    Pooling Layer 3: MaxPooling (2, 2), strides (2, 2).
    Layer 8: 512 filters, (3, 3), ReLU.
    Layer 9: 512 filters, (3, 3), ReLU.
    Layer 10: 512 filters, (3, 3), ReLU.
    Pooling Layer 4: MaxPooling (2, 2), strides (2, 2).
    Layer 11: 512 filters, (3, 3), ReLU.
    Layer 12: 512 filters, (3, 3), ReLU.
    Layer 13: 512 filters, (3, 3), ReLU.
    Pooling Layer 5: MaxPooling (2, 2), strides (2, 2).

- Flattening Layer: Flattens output.

- Dense Layers:
    Fully Connected Layer 1: 4056 units, ReLU.
    Dropout Layer: 0.5 rate.
    Fully Connected Layer 2: 4056 units, ReLU.
    Output Layer: 7 units, softmax.

-Compile the model using optimizer-Adam

- Defining the batch size
- Setting stopEarly
- train the model
- how the show result on pyplot
    -show train and validation train chart
    -show the loss and validation loss chart
- Save the model


## Test the model
- import the library
- load the model
- get the categories from teh model
- find the face inside the image
- we resized the image accordingly
- then resize the rectangle as we only want the face in the image
- then prepare the model for testing 
- add text in the image about what categoty it predicts

