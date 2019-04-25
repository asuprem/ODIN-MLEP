from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

def build_classifier():
    # Initialize CNN
    classifier = Sequential()
    # Add convolutional layer
    #   32 filters
    #   Filter shape: 3 x 3
    #   Input image: 64 x 64 (RGB)
    #   Activation function: RELU
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
    # Add pooling layer
    #   2 x 2 matrix
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Add flattening layer
    classifier.add(Flatten())
    # Add hidden layers (32 nodes)
    classifier.add(Dense(units=32, activation="relu"))
    # Add output layer
    classifier.add(Dense(units=1, activation="softmax"))
    # Compile CNN
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

def train_classifier(classifier):
    datagen = ImageDataGenerator()
    training_set = datagen.flow_from_directory("training_set", target_size=(64, 64), batch_size=32,
            class_mode="binary")
    validation_set = datagen.flow_from_directory("test_set", target_size=(64, 64), batch_size=32,
            class_mode="binary")
    classifier.fit_generator(training_set, steps_per_epoch=686, epochs=4,
            validation_data=validation_set, validation_steps=294)

def main():
    classifier = build_classifier()
    train_classifier(classifier)

if __name__ == "__main__":
    main()
