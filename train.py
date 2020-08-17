from __future__ import print_function

import os
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

from utils import angle_error, RotNetDataGenerator, get_train_train_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_file = 'dataset.npy' # path to data set, can be a npy file or a folder with images.

train_images, test_images = get_train_train_data(dataset_file, test_split_percentage=0.2)

print('Images in train set', train_images.shape[0])
print('Images in test set', test_images.shape[0])


# number of classes
nb_classes = 360
# input image shape
input_shape = (224, 224, 3)

# load base model
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=input_shape)

# append classification layer
x = base_model.output
x = Flatten()(x)
final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=final_output)

model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=[angle_error])

# Load any pre existing weight if required to set as a starting point.
# model.load_weights("model-weight-path")

# training parameters
batch_size = 64
nb_epoch = 30

output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# callbacks
monitor = 'val_angle_error'

checkpointer = ModelCheckpoint(filepath="models/weights_persons.{epoch:02d}-{val_loss:.2f}.hdf5", monitor = "val_loss", verbose = 0,
  save_best_only = True, save_weights_only = False, mode = "min", period = 1)


reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard()

print("Model Training Begin")

# training loop
model.fit(
    RotNetDataGenerator(
        train_images,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_images) / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        test_images,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(test_images) / batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10
)

print("Model Training Finished")