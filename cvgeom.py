
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
import cv2


def pixmap2arr(pixmap:QPixmap):
    image = pixmap.toImage()
    b = image.constBits()
    depth = image.depth() // 8
    
    # sip.voidptr must know size to support python buffer interface
    b.setsize(pixmap.height() * pixmap.width() * depth)
    return np.frombuffer(b, np.uint8).reshape((pixmap.height(), pixmap.width(), depth))

def rgba2hue(arr):
    return cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2HSV)[...,0]


def arr2pixmap(arr:np.ndarray, depth:int=1):
    if depth == 1:
        return QPixmap.fromImage(QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1], QImage.Format.Format_Grayscale8))
    elif depth == 3:
        return QPixmap.fromImage(QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1]*3, QImage.Format.Format_RGB888))
    elif depth == 4:
        return QPixmap.fromImage(QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1]*4, QImage.Format.Format_RGBA8888))
    

def get_edges(arr:np.ndarray):
    gray = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2HSV)[...,0]
    return cv2.Canny(gray, 100, 200)
    

def get_circles(arr:np.ndarray, minRadius=10, maxRadius=200):
    gray = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2HSV)[...,1]
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, minDist=minRadius+maxRadius-10, param1=50, param2=50, 
                               minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        return circles[0]
    else:
        return []

def get_vline(arr:np.ndarray):
    
    gray = cv2.cvtColor(arr[...,:3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    vline = np.argmax(np.sum(edges, axis=0))
    return vline
