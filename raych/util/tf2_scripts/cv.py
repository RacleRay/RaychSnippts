import os
from keras_preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.applications.mobilenet import decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError

import pandas as pd

SOURCE = "./data"

df = pd.read_csv(os.path.join(SOURCE, "train.csv"), na_values=['NA', '?'])
df['filename'] = "clips-" + df["id"].astype(str) + ".jpg"

TRAIN_PCT = 0.9
TRAIN_CUT = int(len(df) * TRAIN_PCT)

df_train = df[0:TRAIN_CUT]
df_validate = df[TRAIN_CUT:]

training_datagen = ImageDataGenerator(rescale=1. / 255,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      fill_mode='nearest')

train_generator = training_datagen.flow_from_dataframe(dataframe=df_train,
                                                       directory=SOURCE,
                                                       x_col="filename",
                                                       y_col="clip_count",
                                                       target_size=(256, 256),
                                                       batch_size=32,
                                                       class_mode='other')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = validation_datagen.flow_from_dataframe(dataframe=df_validate,
                                                       directory=SOURCE,
                                                       x_col="filename",
                                                       y_col="clip_count",
                                                       target_size=(256, 256),
                                                       class_mode='other')

input_tensor = Input(shape=(256, 256, 3))

base_model = ResNet50(include_top=False,
                      weights=None,
                      input_tensor=input_tensor,
                      input_shape=None)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
model = Model(inputs=base_model.input, outputs=Dense(1)(x))
# model.summary()

# Important, calculate a valid step size for the validation dataset
STEP_SIZE_VALID = val_generator.n // val_generator.batch_size

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=[RootMeanSquaredError(name="rmse")])
monitor = EarlyStopping(monitor='val_loss',
                        min_delta=1e-3,
                        patience=50,
                        verbose=1,
                        mode='auto',
                        restore_best_weights=True)
history = model.fit(train_generator,
                    epochs=100,
                    steps_per_epoch=250,
                    validation_data=val_generator,
                    callbacks=[monitor],
                    verbose=1,
                    validation_steps=STEP_SIZE_VALID)
