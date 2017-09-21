import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import cv2
from lesson_functions import *
from moviepy.editor import VideoFileClip
from skimage.feature import hog
from scipy import ndimage as ndi

# ***********************************************************************************************
# loading the features
features=pickle.load(open("features.pickles","rb"))
color_space=features["color_space"]
orient=features["orient"]
pix_per_cell=features["pix_per_cell"]
cell_per_block=features["cell_per_block"]
hog_channel=features["hog_channel"]
spatial_size=features["spatial_size"]
hist_bins=features["hist_bins"]
spatial_feat=features["spatial_feat"]
hist_feat=features["hist_feat"]
hog_feat=features["hog_feat"]
X_scaler=features["X_scaler"]
svc=features["svc"]

print('Model and parameters loaded.')

# ***********************************************************************************************
# function defining the search windows shape
def window_full_list(image):
    # definition of the rectangles
    shapes = [[400,640,128],[400,500,80]]#,[400,600,160],[430,700,160],[460,720,256]]#,[450,650,200]]
    overlap=0.75
    rects=[]
    for i in range(len(shapes)):
        #print (i,shapes[i][0], shapes[i][1])
        rects += slide_window(image, x_start_stop=[None, None], y_start_stop=[shapes[i][0], shapes[i][1]],
                           xy_window=(shapes[i][2], shapes[i][2]), xy_overlap=(overlap, overlap))
        #print(len(rects))
    return rects

# ***********************************************************************************************
# remove comment to draw all boxes on a sample image
#plt.imshow(draw_boxes(mpimg.imread("test_images\\test1.jpg"),window_full_list(mpimg.imread("test_images\\test1.jpg"))))
#plt.show()

# ***********************************************************************************************
# launch of the image detection
def image_detection(img):
    img2 = img.astype(np.float32) / 255
    windows=window_full_list(img)
    on_windows=[]
    on_windows=search_windows(img2, windows, clf=svc, scaler=X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    return on_windows

'''
# ***********************************************************************************************
# testing the detection process through the images in test folder, returning images with boxes

# init the static variable (class)
heatmap.old = np.zeros((720, 1280))

images = glob.glob('test_images\\test*.jpg')
for image in images:
    img=mpimg.imread(image)
    on_windows=image_detection(img)
    img = draw_boxes(img, on_windows)
    threshold=1
    history=0
    hot_windows=combine_boxes(on_windows, img.shape,threshold,history)
    print(len(on_windows), '/', len(window_full_list(img)),'=>',len(hot_windows))
    img=draw_boxes(img, hot_windows,(255,0,0))
    plt.imshow(img)
    plt.show()
'''

'''
# ***********************************************************************************************
# display heat map of a specific image
image='test_images\\test4.jpg'
img=mpimg.imread(image)
on_windows=image_detection(img)
heat=create_heatmap(on_windows,img)
heat = apply_threshold(heat,1)
heatmap = np.clip(heat, 0, 255)
plt.imshow(img)
plt.show()
plt.imshow(heatmap, cmap='hot')
plt.show()
'''


# ***********************************************************************************************
# video pipeline
heatmap.old = np.zeros((720, 1280))

def process_the_image(img):
    # detect boxes
    on_windows = image_detection(img)
    # Combine overlapping windows
    threshold = 0.6
    history = 0.8
    hot_windows = combine_boxes(on_windows, img.shape, threshold, history)
    img = draw_boxes(img, hot_windows, (255, 0, 0))
    img = draw_boxes(img, on_windows, (0, 0, 255),thick=1)
    return img

# init the static variable (class)
heatmap.old=np.ones((720,1280))

# launch of video processing
output = "project_video_h8t6.mp4"
#output = "h8 t7.mp4"
clip1 = VideoFileClip("project_video_with_lines.mp4")
#clip1 = VideoFileClip("test_video.mp4")
clip = clip1.fl_image(process_the_image)
clip.write_videofile(output, audio=False)

