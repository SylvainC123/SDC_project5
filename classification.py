import cv2
import glob
from IPython.display import HTML
from lesson_functions import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
import random
from scipy import ndimage as ndi
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import time

# *****************************************************************************
# Extraction of features
cars = glob.glob('vehicles\*\*.png')
notcars = glob.glob('non-vehicles\*\*.png')

# HOG parameters
color_space = 'HLS'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
# Spatial and color parameters
spatial_size = (16, 16)
hist_bins = 16  # 32

spatial_feat = True
hist_feat = True
hog_feat = True

t = time.time()
car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                hist_bins=hist_bins, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
print("...")
notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size,
                                   hist_bins=hist_bins, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                   spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract HOG features...')

# ***********************************************************************
# Features are centered and normalised
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# **************************************************************************
# Split of the data between training set and testing set

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# ****************************************************************************
# Training of the classifier
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# *****************************************************************************
# Saving the features
features={}
features["color_space"]=color_space
features["orient"]=orient
features["pix_per_cell"]=pix_per_cell
features["cell_per_block"]=cell_per_block
features["hog_channel"]=hog_channel
features["spatial_size"]=spatial_size
features["hist_bins"]=hist_bins
features["spatial_feat"]=spatial_feat
features["hist_feat"]=hist_feat
features["hog_feat"]=hog_feat
features["X_scaler"]=X_scaler
features["svc"]=svc
features["car"]=car_features
features["nocar"]=notcar_features
pickle.dump(features,open("features.pickles","wb"))
print('features saved')