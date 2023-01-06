# ## Data Augmentation
# 
# - 30 classes
# - Augmentation: Randomized skew + shear 
# - Creates 50 additional images per class

# example of horizontal shift image augmentation
from numpy import expand_dims
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import scipy
import pathlib
import PIL
import matplotlib.pyplot as plt
import Augmentor
import shutil

HOME_DIR = "/Users/ruhiyusuf/projects/frc-vision"
WORKING_DIR = os.path.join(HOME_DIR, "modeltraining")
DATA_DIR = os.path.join(WORKING_DIR, "data") # change path accordingly
NUM_CLASSES = 30

CREATE_DATA = False

if CREATE_DATA: 
    # convert png to jpeg
    from PIL import Image
    for i in range(0, NUM_CLASSES):
        im1 = Image.open(os.path.join(DATA_DIR, str(i) + '.png'))
        im1 = im1.convert('RGB')
        im1.save(os.path.join(DATA_DIR, str(i) + '.jpeg'))

if CREATE_DATA:
    # add img to new folders
    for i in range(0, NUM_CLASSES):
        if (not(os.path.exists(os.path.join(DATA_DIR, str(i))))):
            os.mkdir(os.path.join(DATA_DIR, str(i)))
            os.rename(os.path.join(DATA_DIR, str(i) + '.jpeg'), os.path.join(DATA_DIR, str(i), str(i) + '.jpeg'))
    
image_count = len(list(pathlib.Path(DATA_DIR).glob('*/*.jpeg')))
print("Total # of images:", image_count)

id0 = list(pathlib.Path(DATA_DIR).glob('0/*'))
PIL.Image.open(str(id0[0]))

"""
# adding to model architecture
trainAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255),
	preprocessing.RandomFlip("horizontal_and_vertical"),
	preprocessing.RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	preprocessing.RandomRotation(0.3)
])
"""

"""
def augment(images, labels):
	images = tf.image.random_flip_left_right(images)
	images = tf.image.random_flip_up_down(images)
	images = tf.image.rot90(images)

	return (images, labels)

"""

# create data generator
img = load_img(os.path.join(DATA_DIR, '0', '0.jpeg'))

# convert to numpy array
img = img_to_array(img)

def augment_sample(image):
	# image = tf.image.random_flip_left_right(image)
	# image = tf.image.random_flip_up_down(image)
	# image = tf.image.rot90(image)
    image = tf.image.random_jpeg_quality(image, 0, 10)
    return image

def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)


visualize(img, augment_sample(img))
print(img.shape)


if CREATE_DATA:
    for i in range(0, NUM_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(i))
        p = Augmentor.Pipeline(class_dir)

        p.shear(probability=0.40, max_shear_left=20, max_shear_right=20)
        # p.skew_left_right(probability=1.0)
        # p.skew(probability=0.5)
        p.skew(probability=0.9)
        p.sample(50)

        # PIL.Image.open(os.path.join(DATA_DIR, "0", "0.jpeg"))
        for files in os.listdir(os.path.join(class_dir, "output")):
            if files.endswith('.jpeg'):
                shutil.move(os.path.join(class_dir, "output", files), os.path.join(class_dir, files))

if CREATE_DATA:
    for i in range(0, NUM_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(i))

        shutil.rmtree(os.path.join(class_dir, "output"))

# ## Creating Datasets
# - Training
# - Validation
# - Test

batch_size = 32
img_height = 500
img_width = 500

train_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="training",
  seed = 123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

from tensorflow.keras import layers, models
# standardize rgb values (0, 255) to (0, 1)
normalization_layer = tf.keras.layers.Rescaling(1./255)

num_classes = len(class_names)

# sample model
model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=5 
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

