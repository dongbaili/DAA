##################################################
## Project: RotNIST
## Script purpose: To download MNIST dataset and append new rotated digits to it
## tores the images as jpg files with a CSV for labels
## Date: 21st April 2018
## Author: Chaitanya Baweja, Imperial College London
##################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gzip
import os
import numpy as np
import tensorflow as tf
import pandas as pd

import csv
import argparse
from scipy import ndimage
from six.moves import urllib
from PIL import Image
from scipy.misc import imsave

#Url for downloading MNIST dataset
URL = 'http://yann.lecun.com/exdb/mnist/'
#Data Directory where all data is saved
DATA_DIRECTORY = "data"


'''
Download the data from Yann's website, unless it's already here.
filename: filepath to images
Returns path to file
'''

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch sample reweighting experiments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n_train", type=int)
    parser.add_argument("--n_test", type=int)
    parser.add_argument("--test_angle", type=int)
    parser.add_argument("train_angles", nargs="+" , default=None)
    return parser.parse_args()

def download(filename):
    #Check if directory exists
    if not tf.io.gfile.exists(DATA_DIRECTORY):
        tf.io.gfile.makedirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    #Check if file exists, if not download
    if not tf.io.gfile.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(URL + filename, filepath)
        with tf.io.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath
'''
Extract images from given file path into a 3D tensor [image index, y, x].
filename: filepath to images
num: number of images
60000 in case of training
10000 in case of testing
Returns numpy vector
'''
def extract_data(filename, num):
    print('Extracting', filename)
    #unzip data
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, 28, 28,1) #reshape into tensor
    return data

'''
Extract the labels into a vector of int64 label IDs.
filename: filepath to labels
num: number of labels
60000 in case of training
10000 in case of testing
Returns numpy vector
'''
def extract_labels(filename, num):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

'''
Augment training data with rotated digits
images: training images
labels: training labels
'''
def expand_training_data(images, labels, angles):

    expanded_images = []
    expanded_labels = []
    k = 0 # counter
    for x, y in zip(images, labels):
        #print(x.shape)
        k = k+1
        if k%100==0:
            print ('expanding data : %03d / %03d' % (k,np.size(images,0)))

        # register original data
        # expanded_images.append(x)
        # expanded_labels.append(y)

        bg_value = -0.5 # this is regarded as background's value black
        #print(x)
        image = np.reshape(x, (-1, 28))
        #time.sleep(3)
        #print(image)
        #time.sleep(3)
        for angle in angles:# 30 60 90 ... 330
            # rotate the image with random degree
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            #code for saving some of these for visualization purpose only
            image1 = (image*255) + (255 / 2.0)
            new_img1 = (new_img_*255) + (255 / 2.0)
            new_img2 = np.reshape(new_img_,(28,28,1))
            #print(new_img1.shape)

            # register new training data

            expanded_images.append(new_img2)
            expanded_labels.append(y)

    # return them as arrays

    expandedX=np.asarray(expanded_images)
    expandedY=np.asarray(expanded_labels)
    return expandedX, expandedY

def rotate_test_data(images, angle):
    rotated_images = []
    k = 0 # counter
    for x in images:
        #print(x.shape)
        k = k+1
        if k%100==0:
            print ('rotating data : %03d / %03d' % (k,np.size(images,0)))

        bg_value = -0.5 # this is regarded as background's value black
        image = np.reshape(x, (-1, 28))
            
        new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

        # shift the image with random distance
        shift = np.random.randint(-2, 2, 2)
        new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

        #code for saving some of these for visualization purpose only
        new_img2 = np.reshape(new_img_,(28,28,1))
        # register new test data
        rotated_images.append(new_img2)

    return rotated_images

# Prepare MNISt data
def prepare_MNIST_data(angles, use_data_augmentation=True):
    # Get the data.
    train_data_filename = download('train-images-idx3-ubyte.gz')
    train_labels_filename = download('train-labels-idx1-ubyte.gz')
    test_data_filename = download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, args.n_train)
    train_labels = extract_labels(train_labels_filename, args.n_train)
    test_data = extract_data(test_data_filename, args.n_test)
    test_labels = extract_labels(test_labels_filename, args.n_test)

    # expand train data
    if use_data_augmentation:
        train_data,train_labels = expand_training_data(train_data, train_labels, angles)
    # rotate test data
    test_data = rotate_test_data(test_data, args.test_angle)

    if not os.path.isdir("data/train-images"):
        os.makedirs("data/train-images")
    if not os.path.isdir("data/test-images"):
        os.makedirs("data/test-images")
    # process train data
    with open("data/train-labels.csv", 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        writer.writerow(['name','label'])
        for i in range(len(train_data)):
            imsave("data/train-images/" + str(i) + ".jpg", train_data[i][:,:,0])
            writer.writerow([str(i) + ".jpg", train_labels[i]])
    # repeat for test data
    with open("data/test-labels.csv", 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        writer.writerow(['name','label'])
        for i in range(len(test_data)):
            imsave("data/test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
            writer.writerow([str(i) + ".jpg", test_labels[i]])
    #return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels

def prepareTxt(angles, test_angle):
    print(angles)
    df = pd.read_csv("data/train-labels.csv")
    df['id'] = df.index

    df['group'] = df['id'] % len(angles)

    grouped = df.groupby('group')

    # 打印每个分组
    for angle, (name, group) in zip(angles, grouped):

        group_label_counts = group['label'].value_counts().to_dict()
        with open(f"domainbed/txtlist/RMnist/{angle}.txt", 'w') as f:
            # for label, count in group_label_counts.items():
                # filenames = ' '.join(f"data/train-images/{row['name']}" for index, row in group[group['label'] == label].iterrows())
            for index, row in group.iterrows():
                f.write(f"data/train-images/{row['name']} {row['label']}\n")
                
    df= pd.read_csv("data/test-labels.csv")
    group_label_counts = df['label'].value_counts().to_dict()
    angle = test_angle
    with open(f"domainbed/txtlist/RMnist/{angle}.txt", 'w') as f:
        # for label, count in group_label_counts.items():
            # filenames = ' '.join(f"data/test-images/{row['name']}" for index, row in df[df['label'] == label].iterrows())
        for index, row in df.iterrows():
            f.write(f"data/test-images/{row['name']} {row['label']}\n")  

args = get_args()
if(args.train_angles is None):
    args.train_angles = [0, 30 ,60 ,90, 120, 150, 180, 210, 240, 270, 300, 330]
else:
    args.train_angles = [int(x) for x in args.train_angles]

prepare_MNIST_data(args.train_angles, True)

prepareTxt(args.train_angles, args.test_angle)


