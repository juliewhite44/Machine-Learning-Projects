import tensorflow as tf
import numpy as np
from numpy import save

img_size = 64

cat_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "./data/catdata",
    labels="inferred",
    label_mode="int",
    class_names=['catface'],
    color_mode="grayscale",
    batch_size=4,
    image_size=(img_size, img_size),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)

cat_train_labels = []
cat_train_images = []
for images, labels in cat_dataset:
    for i in range(len(images)):
        cat_train_images.append(images[i])
        cat_train_labels.append(labels[i])
c_images = np.array(cat_train_images)
c_images = c_images.reshape(c_images.shape[0], img_size, img_size, )
c_labels = np.array(cat_train_labels)
c_labels = c_labels.reshape(c_labels.shape[0], )

print(c_images.shape)
print(c_labels.shape)
train_labels = c_labels
train_images = c_images

save('./data/images.npy', c_images)
save('./data/labels.npy', c_labels)
