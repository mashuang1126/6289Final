from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical,plot_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from tensorflow.keras import layers,activations,regularizers
from keras import backend as K
from imutils import paths
import keras
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os
import argparse
import random
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

## sample display

image_height = 300
image_width = 300
image_depth = 3
epochs_num = 50
learning_rate = 1e-3
class_num = 2
batch_size = 20
norm_size = 32
decay = learning_rate/epochs_num


input_format = (image_height,image_width,image_depth)

if K.image_data_format() == "channels_first":
    input_format = (image_height,image_width,image_depth)


kernel_size = (3,3)
class_number = 2

model = Sequential(name='CNN')
## Block # 1
model.add(Conv2D(16,kernel_size,
                 # activation='relu',
                 input_shape=input_format,
                 kernel_regularizer= regularizers.l2(0.0001),
                 name='Block_1_Conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),
                           name='Block_1_Pooling'))
# model.add(Dropout(0.25))

## Block # 2
model.add(Conv2D(32,kernel_size,
                     # activation='relu',
                     name='Block_2_Conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),
                           name='Block_2_Pooling'))
# model.add(Dropout(0.25))
## Block # 3
model.add(Conv2D(64,kernel_size,name='Block_3_Conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),
                           name='Block_3_Pooling'))
# model.add(Dropout(0.25))

## Block # 4
model.add(Conv2D(64,kernel_size,padding="same",name='Block_4_Conv2'))
model.add(Activation('relu'))
# model.add(Conv2D(64,kernel_size,
#                      activation='relu',
#                      name='Block_4_Conv2'))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),
#                            name='Block_4_Pooling'))
model.add(MaxPooling2D(pool_size=(2,2),
                           name='Block_4_Pooling'))
# model.add(Dropout(0.25))

## Block# 5
model.add(Conv2D(64,kernel_size,padding="same",
                     # activation='relu',
                     name='Block_5_Conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),
                           name='Block_5_Pooling'))
# model.add(Dropout(0.25))


model.add(Flatten(name='Flatten'))
model.add(Dense(512,name='DenseOne'))
model.add(Activation('relu'))
model.add(Dense(84,name='DenseTwo'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid',name="Prediction"))



############    Model references    ###################
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(300, 300, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
#     tf.keras.layers.MaxPooling2D(2, 2),
#
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.Dense(84, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
######################################################


model.summary()



image_path = "/Users/ShuangMa/Desktop/gender_classification/"
train_path = "/Users/ShuangMa/Desktop/gender_classification/Training/"
validation_path = "/Users/ShuangMa/Desktop/gender_classification/Validation/"

male_training_path = os.path.join(train_path,"male")
female_training_path = os.path.join(train_path,"female")

male_validation_path = os.path.join(validation_path,"male")
female_validation_path = os.path.join(validation_path,"female")

fileNames_male_training = os.listdir(male_training_path)
fileNames_female_training = os.listdir(male_training_path)

fileNames_male_validation = os.listdir(male_validation_path)
fileNames_female_validation = os.listdir(female_validation_path)

print(f"Total Training number of male is : {len(fileNames_male_training)}")
print(f"Total Training number of female is : {len(fileNames_female_training)}")
print(f"Total Validation number of male is : {len(fileNames_male_validation)}")
print(f"Total Validation number of female is : {len(fileNames_female_validation)}")


## sample display
def sample_display(path):
  plt.figure(figsize=(15,15))
  for i in range(8):
    plt.subplot(2,4,i+1)
    image = random.choice(os.listdir(path))
    image = load_img(os.path.join(path,image))
    plt.subplots_adjust(hspace=0.1)
    x = path.split("/")[-1]
    if x == 'male':
      plt.suptitle("Images for Male",fontsize=20)
    else:
      plt.suptitle("Images for Female",fontsize=20)
    plt.imshow(image)
  plt.tight_layout()
  plt.show()

sample_display('/Users/ShuangMa/Desktop/gender_classification/Training/male')
sample_display('/Users/ShuangMa/Desktop/gender_classification/Training/female')

trainingData_generator = ImageDataGenerator(
    # rotation_range = 40,   ####### Use the following parameters when train set is small
    # width_shift_range = 0.2,
    # height_shift_range = 0.2,
    # shear_range = 0.2,
    # zoom_range = 0.2,
    # horizontal_flip = True,
    # vertical_flip = False,
    rescale = 1/255,
    # fill_mode = "nearest", ## "constant","nearest","reflect","warp"
    # data_format = "channels_last"
     ) ## either "channels_last" or "channels_first"

generatorForTrain = trainingData_generator.flow_from_directory(
    image_path + 'Training',
    target_size = (image_height,image_width),
    # color_mode = "rgb",
    class_mode = "binary", ## "binary", default with "categorical", "input","multi_output","raw"
    batch_size = batch_size,
    shuffle = True,
    save_to_dir = '/Users/ShuangMa/Desktop/gender_classification_gen/gen',
    save_prefix = 'trans_',
    save_format = 'jpeg'
    )



# check generated image without fitting the model
generatorForTrain.next()


validationData_generator = ImageDataGenerator(rescale=1/255)

generatorForValidation = validationData_generator.flow_from_directory(
    image_path + 'Validation/',
    target_size = (image_height,image_width),
    batch_size = batch_size,
    class_mode = "binary"
)


## Image for CNN structure
# plot_model(model,
#            to_file='/Users/ShuangMa/Desktop/GWU_Documents/'
#                    'STAT_6289_Statistical_Deep_Learning/Final/model_structure.jpg')

# save_dir = os.path.join(os.getcwd(), 'models_structure')
# model_structure_img = 'model_structure_img.jpg'
#
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_structure_path = os.path.join(save_dir, model_structure_img)
# plot_model(model,
#            to_file=model_structure_path)


## using stochastic gradient decent as optimizer
# opt = SGD(lr=learning_rate,momentum=0.9,decay=decay,nesterov=False)
## Model Compile
# model.compile(loss = 'binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

opt = Adam(lr=learning_rate)

model.compile(loss = 'binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model_fit = model.fit_generator(generatorForTrain,steps_per_epoch=100,epochs=50,
                                   validation_data=generatorForValidation,
                                   validation_steps=50,
                                # callbacks=[early_stopping,history],
                                verbose=1
                                )


# Model performance Visualization

plt.style.use("ggplot")
plt.figure()
# plt.subplot(1,2,1)
plt.plot(model_fit.history['val_accuracy'],label='Validation Accuracy')
plt.title('Model Validation Accuracy')
# plt.ylabel('Validation Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='best')

# plt.subplot(1,2,2)
plt.plot(model_fit.history['val_loss'],label='Validation Loss')
plt.plot(model_fit.history['loss'],label='Training Loss')
plt.plot(model_fit.history['accuracy'],label='Training Accuracy')
plt.title('Training & Testing Loss & Accuracy')
plt.ylabel('Loss & Accuracy')
plt.xlabel('Epochs Number')
plt.legend(loc='best')
save_path = '/Users/ShuangMa/Desktop/GWU_Documents/STAT_6289_Statistical_Deep_Learning/Final/'
plt.savefig(save_path + 'Loss_Accuracy.jpg')
plt.show()

## Save model and weights
model_name = 'gender_class_model.h5'
model_path = os.path.join(save_path, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)



