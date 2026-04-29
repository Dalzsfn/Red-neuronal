import os
from os.path import isfile, join
import numpy as np

def list_files(mnist_path):
    return [join(mnist_path, f) for f in os.listdir(mnist_path) if isfile(join(mnist_path, f))]

def get_images(mnist_path):
    x_train = y_train = x_test = y_test = None
    
    for f in list_files(mnist_path):
        if 'train-images' in f:
            with open(f, 'rb') as data:
                _ = int.from_bytes(data.read(4), 'big')
                num_images = int.from_bytes(data.read(4), 'big')
                rows = int.from_bytes(data.read(4), 'big')
                cols = int.from_bytes(data.read(4), 'big')
                x_train = np.frombuffer(data.read(), dtype=np.uint8).reshape((num_images, rows, cols))
                
        elif 'train-labels' in f:
            with open(f, 'rb') as data:
                data.read(8) 
                y_train = np.frombuffer(data.read(), dtype=np.uint8)
                
        elif 't10k-images' in f:
            with open(f, 'rb') as data:
                _ = int.from_bytes(data.read(4), 'big')
                num_images = int.from_bytes(data.read(4), 'big')
                rows = int.from_bytes(data.read(4), 'big')
                cols = int.from_bytes(data.read(4), 'big')
                x_test = np.frombuffer(data.read(), dtype=np.uint8).reshape((num_images, rows, cols))
                
        elif 't10k-labels' in f:
            with open(f, 'rb') as data:
                data.read(8) 
                y_test = np.frombuffer(data.read(), dtype=np.uint8)
    
    return x_train, y_train, x_test, y_test
