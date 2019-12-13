import cv2
import numpy as np
import requests
import argparse

# url='http://10.16.114.79:8080/shot.jpg'

class VideoReadingError(Exception):
    pass

def open_video(path):
 
    # open video file
    cap = cv2.VideoCapture(path)
    
    while True:
        ret, img = cap.read()
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def open_web_cam():
    
    # open web cam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, img = cap.read()
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def open_web_ip(path):
    while True:
        img_res = requests.get(path)#"http://192.168.0.103:8080/shot.jpg")
        img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
        img = cv2.imdecode(img_arr,-1)
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
        open_web_ip(args.path)
    else:
        raise VideoReadingError("Wrong mode parameter")
