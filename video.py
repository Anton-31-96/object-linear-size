import cv2
import numpy as np
import requests
import argparse
# from pipeline import *
from utils_cv import *
from skimage.transform import rescale 
from skimage import img_as_ubyte

# url='http://10.16.114.79:8080/shot.jpg'
# url

class VideoReadingError(Exception):
    pass

def open_video(path):
    """
    Opens video file at path
    """
    
    # open video file
    cap = cv2.VideoCapture(path)
    j = 0 # j helps to reduce fps on the video
    while(cap.isOpened()):
        
        if j == 0:
            ret, img = cap.read()
            img = np.asarray(img, dtype=np.uint8)
#             img = rescale(img, 0.4)
            try:
                img = pipeline(img)
            except: 
                pass
        elif j > 20:
            j = 0
        else:
            j += 1
     
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything if job is finished
    cv2.destroyAllWindows()

def open_web_cam():
    """
    Opens video from internal web camera of laptop
    """
    # open web cam
    cap = cv2.VideoCapture(0)
    
    while(cap.isOpened()):
        ret, img = cap.read()
        
        img = rescale(img, 0.4, multichannel=True)
        img = img_as_ubyte(img)

        try:
            img = pipeline(img)
        except: 
            pass
        
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything if job is finished
    cv2.destroyAllWindows()

def open_web_ip(path):
    """
    Opens video from ip web camera
    Arguments:
        path (str) - link to the web camera
             example: "http://192.168.0.103:8080/shot.jpg"
    Returns: None
    """
    
    while True:
        img_res = requests.get(path)
        img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
#         img_arr = rescale(img_arr, 0.5, multichannel=True)
        img = cv2.imdecode(img_arr,-1)
        # img = rescale(img, 0.5, multichannel=True)
#         img = rescale(img, 0.8, multichannel=True)
#         img = img_as_ubyte(img)
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
#         args.path ='http://192.168.1.64:8080/shot.jpg'
        open_web_ip(args.path)
    else:
        raise VideoReadingError("Wrong mode parameter")
