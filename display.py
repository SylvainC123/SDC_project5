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

# ************************************************************

# Read in car and non-car images and display some sample and characteristics
path=('vehicles\\GTI_Far\\','vehicles\\GTI_Left\\','vehicles\\GTI_MiddleClose\\','vehicles\\GTI_Right\\','vehicles\\KITTI_extracted\\',
     'non-vehicles\\Extras\\','non-vehicles\\GTI\\')

def plot_random_samples(path):
    image_per_set=10
    plt.figure(1)
    for i in range(len(path)):
        images = glob.glob(path[i]+'*.png')
        print('directory :',path[i])
        print('image shape:',(mpimg.imread(images[0])).shape)
        print('image number:',len(images))
        # select images at random
        plt.figure(figsize=(11,10))
        for j in range(image_per_set):
            k=random.choice(images)
            img=mpimg.imread(k)
            plt.subplot(7,image_per_set,j+1)
            fig = plt.imshow(img)
            fig.set_cmap('hot')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        plt.show()

plot_random_samples(path)

# ****************************************************************
# check proportion of car and nocars
car_images = glob.glob('vehicles/*/*.png')
noncar_images = glob.glob('non-vehicles/*/*.png')
print(len(car_images), len(noncar_images))

# *****************************************************************

cars= glob.glob('vehicles\*\*.png')
notcars = glob.glob('non-vehicles\*\*.png')

image = mpimg.imread(random.choice(cars))
plt.imshow(image)

# Define a function to compute color histogram features
def color_hist2(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0]*255, bins=32, range=(0, 256))
    ghist = np.histogram(img[:, :, 1]*255, bins=32, range=(0, 256))
    bhist = np.histogram(img[:, :, 2]*255, bins=32, range=(0, 256))
    # Generating bin centers
    bin_centers = (rhist[1][1:] + rhist[1][0:len(rhist[1]) - 1]) / 2.
    #plt.bar(bin_centers, rhist[0])
    plt.show()
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[1], ghist[1], bhist[1]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features

rh, gh, bh, bincen, feature_vec = color_hist2(image, nbins=32, bins_range=(0, 256))
# Plot a figure with all three bar charts
if rh is not None:
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.bar(bincen, rh[0])
    plt.title('R Histogram')
    plt.subplot(132)
    plt.bar(bincen, gh[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(133)
    plt.bar(bincen, bh[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')
    fig.tight_layout()
    plt.show()
else:
    print('Your function is returning None for at least one variable...')

# *******************************************************************


