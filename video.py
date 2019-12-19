import cv2
import numpy as np
import requests
import argparse
from pipeline import *
from skimage.transform import rescale 

# url='http://10.16.114.79:8080/shot.jpg'

class VideoReadingError(Exception):
    pass

def open_video(path):
    """
    Opens video file at path
    """
    # open video file
    cap = cv2.VideoCapture(path)
    j = 0
    while(cap.isOpened()):
        j+=1
        ret, img = cap.read()
#         img = rescale(img, 0.4)
        if j % 10 == 0:
            img = np.asarray(img, dtype=np.uint8)
            try:
                img = pipeline(img)
            except: 
                pass
            j = 0
     
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def open_web_cam():
    """
    Opens video from internal web camera of laptop
    """
    # open web cam
    cap = cv2.VideoCapture(0)
    
    while(cap.isOpened()):
        ret, img = cap.read()
                
        try:
            img = pipeline(img)
        except: 
            pass
        
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def open_web_ip(path):
    """
    Opens video from ip web camera
    Arguments:
        path (str) - link to the web camera
             example: "http://192.168.0.103:8080/shot.jpg"
    Returns: None
    """
    while True:
        img_res = requests.get(path)#
        img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
        img = cv2.imdecode(img_arr,-1)
                
        try:
            img = pipeline(img)
        except: 
            pass
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str,
                        help="write a mode for reading videfile mode = ('webcam' | 'video' | 'web_ip')")
    parser.add_argument("--path", action='store',
                        help="path to vide file")
    args = parser.parse_args()
    
    mode = args.mode
    
    if mode == 'webcam':
        open_web_cam()
    elif mode == 'video':
        open_video(args.path)
    elif mode == 'web_ip':
        args.path ='http://192.168.1.64:8080/shot.jpg'
        open_web_ip(args.path)
    else:
        raise VideoReadingError("Wrong mode parameter")
