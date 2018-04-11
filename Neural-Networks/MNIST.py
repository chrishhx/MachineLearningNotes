# -*- coding: utf-8 -*-
"""
This is a script file intend to read MNIST dataset.
"""

import numpy as np

def read_data(label_file,img_file,dummy=True):
    with open(label_file,'rb') as file:
        magic_number = int.from_bytes(file.read(4),byteorder='big');
        if magic_number == 2049:
            Ncases = int.from_bytes(file.read(4),byteorder='big');
            buf = file.read()
            if (dummy):
                index = np.frombuffer(buf,dtype=np.uint8).reshape(Ncases);
                labels = np.zeros(shape=(Ncases,10),dtype=bool);
                labels[range(Ncases),index] = True;
            else:
                labels = np.frombuffer(buf,dtype=np.uint8).reshape(Ncases);
        else:
            print('label file corrupted');
    
    with open(img_file,'rb') as file:
        magic_number = int.from_bytes(file.read(4),byteorder='big');
        if magic_number == 2051:
            Ncases = int.from_bytes(file.read(4),byteorder='big');
            rows = int.from_bytes(file.read(4),byteorder='big');
            cols = int.from_bytes(file.read(4),byteorder='big');
            buf = file.read();
            images = np.frombuffer(buf,dtype=np.uint8).reshape(Ncases,rows,cols);
        else:
            print('image file corrupted');
    return labels,images

import os
os.chdir('d:/workspace')
train_labels,train_images = read_data('train-labels.idx1-ubyte','train-images.idx3-ubyte')

''' check by imshow
import matplotlib.pyplot as plt
print(np.where(train_labels[0:3])[1])
plt.imshow(train_images[0],cmap='gray')
plt.imshow(train_images[1],cmap='gray')
plt.imshow(train_images[2],cmap='gray')
'''
