# Convolutional Neural Network

# Example: Is this a Cat or Dog

# Data folder structure:
#                           data:
#                               training_set:
#                                       cats
#                                       dogs
#                               test_set:
#                                       cats
#                                       dogs
#                               single_prediction:

# Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # Initalise NN
from keras.layers import Conv2D # to add convolution layer
from keras.layers import MaxPooling2D # Max pooking layers
from keras.layers import Flatten # to convert pooled layer to feature vector for input
from keras.layers import Dense # Add fully connected layer to ANN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
    # 32 feature ditectors/filters
    # 3x3 kernel_size
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Max Pooling
    # 2x2 to reduse size withou losing too much info
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer to inprove accuracy of this praticular set
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
    # Add first hidden layer
    # Units = Some whare between input and output nodes, power of 2
classifier.add(Dense(units = 128, activation = 'relu'))
    # Output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
    # Image augmintatin - transform/resize/crop to stop over fitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=(8000/32),
                         epochs=25,
                         validation_data=test_set,
                         validation_steps= (2000/32))

# Making a single new prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)














