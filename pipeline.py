from utils_cv import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import collections
import time
from skimage import io
from skimage.feature import canny
from skimage.transform import rescale, resize
from skimage.morphology import dilation, disk
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import time
from IPython import display
from scipy.ndimage import distance_transform_edt



def normalize_image(image, image_colored, rescale_param = 0.5):
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
    image_warped = warp(image_colored, tform, output_shape=(720,1020))#[:720,:1020]

    data = image_warped.astype('float64') / np.max(image_warped)
    data = 255 * data
    img = data.astype('uint8')
    img = adjusting_brightness(img[30:-5, 15:-15], a = 1.5, b = 3) # notice that slice is a KOSTYL here
    return img, tform

def pipeline(original):
    def middle(x, y):
        return ((x[0]+y[0])/2, (x[1]+y[1])/2)
    
    image_redone, tform = normalize_image(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY),
                                   original,
                                   rescale_param=0.5)
    
    gray = cv2.cvtColor(image_redone,
                         cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    img, ext_contours, hierarchy = cv2.findContours(edged.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

    (cnts, _) = contours.sort_contours(ext_contours)
    cnts = [cnt[:, 0] for cnt in cnts]
    pixelsPerMetric = None

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

        dimA = distance_2 * 0.076
        dimB = distance_1 * 0.076
        dists.append((dimA, dimB))
        
        box = tform.__call__(box)
        boxes.append(box.astype('int32'))

#     result = np.pad(result, ((30,5), (15, 15), (0,0)), mode='constant') #pading is the same KOSTYL as before
#     result = warp(result, tform.inverse, output_shape=original.shape)
#     result = result.astype('uint8')
#     mask = np.nonzero(result[:,:,0])

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
        cv2.putText(original, "{:.1f} cm".format(dimA),
            (int(top_x - 10), int(top_y - 10)), 0, 0.65, (255, 0, 0), 2)

        cv2.circle(original, (int(right_x), int(right_y)), 5, (255, 0, 0), -1)
        cv2.putText(original, "{:.1f} cm".format(dimB),
            (int(right_x + 5), int(right_y)), 0, 0.65, (255, 0, 0), 2)
    
#     original[mask] = [200, 0, 0]
    
    return original
