import cv2
import os
import numpy as np

#holdtime

os.makedirs('holdtime',exist_ok=True)

for i in range(0, 450):
    img_kukeis=[]
    img0 = cv2.imread('triming/th/th_%04d.bmp' %i)
    img1 = cv2.imread('triming/th/th_%04d.bmp' %(i+1))
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    nLabels_0, labeled_0, data_0, center_0 = cv2.connectedComponentsWithStats(gray0)
    nLabels_1, labeled_1, data_1, center_1 = cv2.connectedComponentsWithStats(gray1)
    center=[]
    for n in range(nLabels_0): 
        # print(center_1[n])
        center.append(np.append(center_0[n],0))
    for k in range(1,nLabels_0):
        for l in range(1,nLabels_1):
            if ((center[k][0]-center_1[l][0])**2 - (center[k][1]-center_1[l][1])**2)**0.5 <= 0.2:
                center[k][2] += 1
        if center[k][2] == 1:
            x, y, w, h, size = data_0[k,0],data_0[k,1],data_0[k,2],data_0[k,3],data_0[k,4]
            img_kukei = cv2.rectangle(img0,(x-1,y-1),(x+w,y+h),(0,255,0),1)


    cv2.imwrite('holdtime/image_kukei-%04d.bmp' % i, img_kukei)
