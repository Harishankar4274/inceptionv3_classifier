import json
import PIL
from keras.applications import InceptionV3 as inceptionv3

img_rows = 299
img_cols = 299

model = inceptionv3(weights = 'imagenet',
                   include_top = False,
                   input_shape = (img_rows,img_cols,3))

for layer in model.layers:
    layer.trainable = False

def addTopModel(bottom_model, num_class, D=256):
    top_model = bottom_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(D, activation="relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation="softmax")(top_model)
    return top_model


from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

file = open('config.json')
data = json.load(file)

num_classes = int(data['num_classes'])

FC_Head = addTopModel(model, num_classes)

new_model = Model(inputs=model.input, outputs=FC_Head)

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = data['train_data_dir']
val_data_dir = data['val_data_dir']

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_batchsize = data['train_batchsize']
val_batchsize = data['val_batchsize']

train_gen = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=train_batchsize,
    class_mode='categorical')

val_gen = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

chk_pt = data['ModelCheckPoint']
checkpoint = ModelCheckpoint('/dataset/model.hd5',
                             monitor=chk_pt['monitor'],
                             mode=chk_pt['mode'],
                             save_best_only=chk_pt['save_best_only'],
                             verbose=chk_pt['verbose'])

es = data['EarlyStopping']
earlystop = EarlyStopping(monitor=es['monitor'],
                          min_delta=es['min_delta'],
                          patience=es['patience'],
                          verbose=es['verbose'],
                          restore_best_weights=es['restore_best_weights'])

callbacks = [earlystop, checkpoint]

new_model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

train_samples = data['train_samples']
val_samples = data['validation_samples']
epochs = data['num_epochs']
batch_size = data['batch_size']

history = new_model.fit_generator(
    train_gen,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_gen,
    validation_steps=val_samples // batch_size)
new_model.save('/dataset/model.hd5')