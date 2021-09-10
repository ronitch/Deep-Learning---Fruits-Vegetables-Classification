import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
import pickle
import cv2
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from IPython.display import display 
from PIL import Image, ImageOps
   

model1 = load_model('my_model.h5')
 

from sklearn.datasets import load_files       #Load text files with categories as subfolder names
import numpy as np

train_dir = "D:/Ronit/Data Science/Projects/Deep Learning - Image Recognition/Data/fruits-360/Training"
test_dir = "D:/Ronit/Data Science/Projects/Deep Learning - Image Recognition/Data/fruits-360/Test"

def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
    
x_train, y_train,target_labels = load_dataset(train_dir)
x_test, y_test,_ = load_dataset(test_dir)
print('Loading complete!')       

st.write("Fruits and Vegetables Classification")
file = st.file_uploader("Please upload an image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    size = (100,100)
    image1 = ImageOps.fit(image, size, Image.ANTIALIAS)
    image2 = np.asarray(image1)
    img_reshape = image2[np.newaxis,...]
    prediction = model1.predict(img_reshape)
    score = tf.nn.softmax(prediction[0])
    st.write("This image most likely belongs to {} with a {:.2f} percent confidence.".format(target_labels[np.argmax(score)], 100 * np.max(score)))
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(target_labels[np.argmax(score)], 100 * np.max(score))
)