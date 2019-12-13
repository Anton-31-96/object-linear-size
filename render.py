import cv2 
import numpy as np
import matplotlib.pyplot as plt

def render_text(img, text, org):
    """
    returns image with text on it
    """
    
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 

    # fontScale 
    fontScale = 1

    # Blue color in BGR 
    color = (255, 0, 0) 

    # Line thickness of 2 px 
    thickness = 2
    
    cv2.putText(img, text, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA)
    
    return img