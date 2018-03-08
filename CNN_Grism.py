import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


# Importing Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


## Part 1 - Building CNN
# Initializing an CNN
classifier = Sequential()

# Step 1: Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
''' Convolution2D(no. of filters, size of filter, input_size = (Coloured/blackwhite)) 
No of filters can be increased when worked on GPU
input_size = no. more if GPU
Activition = relu
'''


# Step 2: MaxPooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding another Convolution Layer (To improve the accuracy)
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# to improve the efficency we can add one more convolution layer or one more Full connection.
# (without changing the parameters)


# Step 3: Flatten
classifier.add(Flatten())

# Step 4: Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu')) #(power of 2, common practice)
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
'''
output_dim value can be increased if GPU
more than 2 outcomes activation should be softax, instead of sigmoid
'''

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Loss to categorical_crossentropy

## Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        # This should be same as input_shape.
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

'''
To get higher accuracy take more target set
'''

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
'''
steps_per_epoch = # of training set
epochs = more if GPU
validation_step = # of test set
'''