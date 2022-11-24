import cv2
import os
import numpy as np

readfile = 'triming2'
writefile = 'holdtime_sabun_1'

os.makedirs(writefile,exist_ok=True)

for i in range(0, 100):
    img_kukeis=[]
    img0 = cv2.imread(readfile + '/th/th_%04d.bmp' %i)
    img1 = cv2.imread(readfile + '/th/th_%04d.bmp' %(i+1))
    img2 = cv2.imread(readfile + '/th/th_%04d.bmp' %(i+2))
    img3 = cv2.imread(readfile + '/th/th_%04d.bmp' %(i+3))
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    bitwise_and_1 = cv2.bitwise_and(gray0, gray1)
    bitwise_and_2 = cv2.bitwise_and(gray0, gray2)
    bitwise_and_3 = cv2.bitwise_and(gray0, gray3)
    nLabels_0, labeled_0, data_0, center_0 = cv2.connectedComponentsWithStats(gray0)
    nLabels_1, labeled_1, data_1, center_1 = cv2.connectedComponentsWithStats(bitwise_and_1)
    nLabels_2, labeled_2, data_2, center_2 = cv2.connectedComponentsWithStats(bitwise_and_2)
    nLabels_3, labeled_3, data_3, center_3 = cv2.connectedComponentsWithStats(bitwise_and_3)
    center=[]
    for n in range(nLabels_0): 
        # print(center_1[n])
        center.append(np.append(center_0[n],0))
    for k in range(1,nLabels_0):
        for l in range(1,nLabels_1):
            if ((center[k][0]-center_1[l][0])**2 - (center[k][1]-center_1[l][1])**2)**0.5 <= 1:
                for m in range(1,nLabels_2):
                    if ((center[k][0]-center_2[m][0])**2 - (center[k][1]-center_2[m][1])**2)**0.5 <= 1:
                        for n in range(1,nLabels_3):
                            if ((center[k][0]-center_3[n][0])**2 - (center[k][1]-center_3[n][1])**2)**0.5 <= 1:
                                center[k][2] += 1
        if center[k][2] == 1:
            x, y, w, h, size = data_0[k,0],data_0[k,1],data_0[k,2],data_0[k,3],data_0[k,4]
            img_kukei = cv2.rectangle(img0,(x-1,y-1),(x+w,y+h),(0,255,0),1)


    cv2.imwrite(writefile + '/image_kukei-%04d.bmp' % i, img_kukei)
