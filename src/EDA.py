'''
Exploratory Data Analysis on the training data. This includes:

A. square have the following info: size, color, position, background, construct the following plots:

x-axis: size, y-axis: H, S, V, R, G, B, 

a: blue, b: green, c: red

B. check if the sqare has pure color, or has more information, by checking the entropy distribution for each image
'''
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data import SquareDataset
import torch.utils.data as torchdata
import cv2
import numpy as np
import skimage.measure as measure
from tqdm.auto import tqdm

# %%
def feature_engineering(dataset):
    '''
    compute the dataset's rgb distribution'
    '''
    individual_ = {
        'r': [],
        'g': [],
        'b': [],
        'h': [],
        's': [],
        'v': [],
        'size': [],
    }
    info = {
        'a': individual_.copy(), 
        'b': individual_.copy(),
        'c': individual_.copy(),
    }
    label_mapping = {
        0: 'a',
        1: 'b',
        2: 'c',
    }
    for idx, el in enumerate(tqdm(dataset)):
        img = el[0].numpy()
        img_hsv = cv2.cvtColor((img.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        img_hsv = img_hsv / 255
        label = label_mapping[el[1]]   
        roi = (img[0, :, :] < 0.99) & (img[1, :, :] < 0.99) & (img[2, :, :] < 0.99)
        img = img[:, roi]
        
        info[label]['r'].append(np.mean(img[0, :]))
        info[label]['g'].append(np.mean(img[1, :]))
        info[label]['b'].append(np.mean(img[2, :]))
        info[label]['h'].append(np.mean(img_hsv[:, :, 0]))
        info[label]['s'].append(np.mean(img_hsv[:, :, 1]))
        info[label]['v'].append(np.mean(img_hsv[:, :, 2]))
        info[label]['size'].append(roi.sum())
    return info
        
        
                         
# %%
dataset = SquareDataset('/home/fang/Square_Task/squares/train')
info = feature_engineering(dataset)

# %%
# draw several images
iter_ = iter(dataset)
for i in range(5):
    img, label = next(iter_)
    img = img.numpy()
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    plt.subplot(1, 5, i+1)
    plt.imshow((img < (0.99 * 255)).astype(np.float32))
    plt.axis('off')
    plt.title(f'label: {label}')

# %%
# plot each class's rgb and size distribution
for cls_, individual_ in info.items():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (k, v) in enumerate(individual_.items()):
        if k == 'size':
            continue
        ax = axes[i//3, i%3]
        sns.histplot(v, ax=ax)
        ax.set_title(f'class {cls_}: {k}')
    plt.show()

# %%
# based on the above plots, we can see that we'd better use the size, r, g, b as the features.
all_r = info['a']['r'] + info['b']['r'] + info['c']['r']
all_g = info['a']['g'] + info['b']['g'] + info['c']['g']
all_b = info['a']['b'] + info['b']['b'] + info['c']['b']
mean_r = np.mean(all_r)
mean_g = np.mean(all_g)
mean_b = np.mean(all_b)
std_r = np.std(all_r)
std_g = np.std(all_g)
std_b = np.std(all_b)
print(f'mean_r: {mean_r}, mean_g: {mean_g}, mean_b: {mean_b}')
print(f'std_r: {std_r}, std_g: {std_g}, std_b: {std_b}')

# %%
# try a random forest model
x = []
y = []
for cls_, individual_ in info.items():
    for i in range(len(individual_['r'])):
        x.append([individual_['r'][i], individual_['g'][i], individual_['b'][i], individual_['size'][i]])
        y.append(cls_)
x = np.array(x)
y = np.array(y)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy: {accuracy}')

# %%
