import cv2
import time
import numpy as np

bbox = (720, 220, 80, 60)
frame1 = cv2.imread('frame1.png')
frame2 = cv2.imread('frame2.png')
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

tracker = cv2.legacy.TrackerMOSSE_create()
t0 = time.perf_counter_ns()
ok = tracker.init(gray1, bbox)
print(f'Init time: {time.perf_counter_ns() - t0} ns')

t1 = time.perf_counter_ns()
ok, new_bbox = tracker.update(gray2)
print(f'Update time: {time.perf_counter_ns() - t1} ns')
print('Tracking success:', ok)
print('New bbox:', new_bbox)

if ok:
    x, y, w, h = map(int, new_bbox)
    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    cv2.putText(frame2, "Tracking failure", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

cv2.imshow('Results', frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()