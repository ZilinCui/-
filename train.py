from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf

print(tf.__version__)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "./data/train"
test_dir = "./data/test"

train_generator = train_datagen.flow_from_directory(
   train_dir,
   target_size=(64, 64),
   batch_size=20,
   class_mode="binary"
)

test_generator = train_datagen.flow_from_directory(
   test_dir,
   target_size=(64, 64),
   batch_size=20,
   class_mode="binary"
)


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=["accuracy"])



model.fit_generator(train_generator, steps_per_epoch=20 ,epochs=100, verbose=1)
test_loss, test_acc = model.evaluate_generator(test_generator)
print(test_loss, test_acc)