import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.transform import ProjectiveTransform, warp
from skimage import io
from skimage.feature import canny, match_template
from skimage.transform import rescale
from skimage.morphology import dilation, disk
import cv2
import pickle

from skimage import img_as_ubyte

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import collections
import time
from skimage import io
from skimage.feature import canny
from skimage.transform import rescale
from skimage.morphology import dilation, disk
import scipy 
import skimage



from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils

from tqdm import tqdm_notebook as tqdm

import time

from scipy.ndimage import distance_transform_edt

def save_pkl(variable, name):
    name = name + '.pkl'
    output = open(name, 'wb')
    pickle.dump(variable, output)
    output.close()

def load_pkl(name):
    name = name + '.pkl'
    pkl_file = open(name, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    return result

def show(image):
    plt.figure(figsize = (7, 7), dpi = 100)
    plt.imshow(image, cmap = 'gray')
    plt.show()
    
def read_images(array_of_paths):
    array_of_images = []
    for img_path in array_of_paths:
        image = cv2.imread(img_path)
        array_of_images.append(image)
    return array_of_images
    
def print_images(array_of_images, nrows, ncols):
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20, 20))
    for ax, image in zip(axes.flat, array_of_images):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
    plt.tight_layout()
    plt.show()


def divide_into_boxes(image):
    box_size = int(image.shape[0]/9)
    return [cv2.resize(image[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size], (100, 100)) for i in range(9) for j in range(9)]

def adjusting_brightness(image, a = 1.1, b = 0):
    return cv2.convertScaleAbs(image, alpha=a, beta=b)

def normalize_image(image, image_colored, rescale_param = 0.5):
    image = image.copy()
    image_colored = image_colored.copy()
    image_scaled = rescale(image, rescale_param)
    edges = canny(image_scaled)
    
    selem = disk(1)
    edges = dilation(edges, selem)
    
    edges = (edges).astype(np.uint8)
    img, ext_contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(ext_contours, key=cv2.contourArea)    
    contour = contour.squeeze()
    
    epsilon = 0.05 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True).squeeze()
    corners = (perspective.order_points(corners))
    corners = corners/rescale_param

    size_square = min(image_scaled.shape)
    tform = ProjectiveTransform()
    tform.estimate(np.array([[0,0], [1020,0], [1020,720], [0, 720]]), corners)
    image_warped = warp(image_colored, tform)[:720,:1020]

    img = img_as_ubyte(image_warped)
    img = adjusting_brightness(img[30:-5, 15:-15], a = 1.7, b = 3)
    return img, tform

def pipeline(original):
    original = original.copy()
    def middle(x, y):
        return ((x[0]+y[0])/2, (x[1]+y[1])/2)
    
    try:
        image_redone, tform = normalize_image(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY),
                                   original,
                                   rescale_param = 0.6)
    except:
        return None
    
    if type(image_redone) == type(None):
        return None
    
    
    result = image_redone.copy()
    
    median_filtered = scipy.ndimage.median_filter(cv2.cvtColor(image_redone, cv2.COLOR_RGB2GRAY ), size=3)

    threshold = skimage.filters.threshold_otsu(median_filtered)
    predicted = np.uint8(median_filtered > 220) * 255
    gray = cv2.GaussianBlur(median_filtered, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    img, ext_contours, hierarchy = cv2.findContours(edged.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

    (cnts, _) = contours.sort_contours(ext_contours)
    cnts = [cnt[:, 0] for cnt in cnts]
    pixelsPerMetric = 53/image_redone.shape[0]
    result = np.zeros(image_redone.shape)
    boxes = []
    dists = []
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        box = cv2.boxPoints(cv2.minAreaRect(c)).astype('int32')
        box = box + np.array([15,30]) # the same KOSTYL as before
        box = perspective.order_points(box).astype('int32')
       
        top_left, top_right, bottom_right, bottom_left = box

        top_x, top_y = middle(top_left, top_right)
        bottom_x, bottom_y = middle(bottom_left, bottom_right)
        left_x, left_y = middle(top_left, bottom_left)
        right_x, right_y = middle(top_right, bottom_right)

        distance_1 = dist.euclidean((top_x, top_y), (bottom_x, bottom_y))
        distance_2 = dist.euclidean((left_x, left_y), (right_x, right_y))

        dimA = distance_2 * pixelsPerMetric
        dimB = distance_1 * pixelsPerMetric
        dists.append((dimA, dimB))
        
        box = tform.__call__(box)
        boxes.append(box.astype('int32'))

    # render borders on original image 
    for box, dst in zip(boxes, dists):
        cv2.drawContours(original, [box], -1, (255, 0, 116), 2)
        top_left, top_right, bottom_right, bottom_left = box

        top_x, top_y = middle(top_left, top_right)
        bottom_x, bottom_y = middle(bottom_left, bottom_right)
        left_x, left_y = middle(top_left, bottom_left)
        right_x, right_y = middle(top_right, bottom_right)

        dimA = dst[0]
        dimB = dst[1]

        cv2.circle(original, (int(top_x), int(top_y)), 5, (255, 0, 0), -1)
        cv2.circle(original, (int(right_x), int(right_y)), 5, (255, 0, 0), -1)
        
        cv2.putText(original, "{:.1f}cm".format(dimA),
            (int(top_x - 10), int(top_y - 10)), 0, 0.65, (255, 255, 0), 2)
        cv2.putText(original, "{:.1f}cm".format(dimB),
            (int(right_x + 5), int(right_y)), 0, 0.65, (255, 255, 0), 2)
        
    return original
