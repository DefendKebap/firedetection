import mss
import numpy as np
import cv2
import pyautogui
import time
import keyboard
import torch


model = torch.hub.load(r'D:\yolov5-master', 'custom', path=r'D:\yolov5-master\atesbest.pt', source='local')
with mss.mss() as sct:
    monitor = {'top':20, 'left':0, 'width': 860, 'height': 1064}
while True:
    t= time.time()
    img = np.array(sct.grab(monitor))
    result = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size=400)

    cv2.imshow('s', np.squeeze(result.render()))

    print('fps: {}'.format(1/(time.time()-t)))
    cv2.waitKey(1)
    if keyboard.is_pressed('q'):
        break
cv2.destroyAllWindows()


