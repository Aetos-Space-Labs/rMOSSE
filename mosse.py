import cv2
import time
import numpy as np

image = np.zeros((800, 800), dtype=np.uint8)
bbox  = (80, 80, 640, 640)

# warm-up
tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(image, bbox)

t0 = time.perf_counter()
tracker = cv2.legacy.TrackerMOSSE_create()
tracker.init(image, bbox)
print(f"init: {(time.perf_counter() - t0)*1e3:.3f} ms")

# measure first update
t1 = time.perf_counter()
_ = tracker.update(image)
print(f"update: {(time.perf_counter() - t1)*1e3:.3f} ms")
