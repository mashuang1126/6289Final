from keras.preprocessing.image import img_to_array,image
from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
import imutils
import cv2
import os



path = '/Users/ShuangMa/Desktop/GWU_Documents/STAT_6289_Statistical_Deep_Learning/Final/'
model = load_model(path+'no_padding_no_strides.h5')
# model.summary()


image_for_prediction = path+'For_Classification/'


test = os.listdir(image_for_prediction)
print(test)
for test_images in test:
    testimage = os.path.join(image_for_prediction,test_images)
    print(testimage)

images = []
for test_images in test:
    if test_images.endswith(('png','.jpg','.jpeg')):
        fd = os.path.join(image_for_prediction,test_images)
        print(fd)
        images.append(fd)

# random.shuffle(images)
pre_x =[]
for i in images:
    original = cv2.imread(i)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(300,300))
    image = image.astype("float")/255
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    pred = model.predict(image)
    # pred = pred.argmax(axis=1)[0]
    label = "Female" if pred <= 0.5 else "Male"
    color = (0,0,255) if pred <= 0.5 else (0,255,0)

    original = cv2.resize(original,(300,300))
    cv2.putText(original,label,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    pre_x.append(original)
montage = imutils.build_montages(pre_x,(128,128),(4,4))[0]
cv2.imshow("Results",montage)
cv2.waitKey(0)
