import cv2
import time
import numpy as np

image = np.zeros((800, 800), dtype=np.uint8)
bbox = (80, 80, 640, 640)

tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(image, bbox)

timer = time.time()
tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(image, bbox)
print(time.time() - timer)
