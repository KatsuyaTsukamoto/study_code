# -*- coding: utf-8 -*-
"""
Created on Fri octo  23 17:46:49 2020

@author: tsuka
"""

'''---------モジュールのインポート--------'''
import numpy as np
import matplotlib.pylab as plt
import math
import os
from tqdm import tqdm
import cv2
import datetime

'''-------------変更箇所---------------'''
readfolder = 'triming'+'/'+'th' #読み込む画像フォルダ
diameter_max = 30 #最大の液滴の大きさ[pix]
diameter_min = 1 #最小の液滴の大きさ[pix]
frame_begin = 1 #最初の読み込む画像
frame_end = 450  #最後の読み込む画像
fps_shut = 10000 #撮影時の撮影速度
rmax=2.0       ### 探査領域最大値
rmin=0.0       ### 探査領域最小値
re=2        ### 探査領域2回目以降
xscale=0.0034  ### 1ピクセルの長さ[cm]x方向
yscale=0.0034  ### 1ピクセルの長さ[cm]y方向
velrange_min=0 #速度のグラフの最小値
velrange_max=0.5 #速度のグラフの最大値
scales=[xscale,yscale]
scales=np.array(scales)
time=np.float(1.0/fps_shut)

'''------------関数定義--------------------'''
'''<<<<<<<<<メイン関数>>>>>>>>>>>>'''
def main(date):

    '''___________フォルダ作成_____________'''
    ptv="/ptv"
    os.makedirs(date+ptv, exist_ok=True)#resultフォルダ作成
    vel_hists="/vel_hist"
    os.makedirs(date+ptv+vel_hists, exist_ok=True)#dias_histフォルダ作成
    '''-------------解析-------------------'''
    i2=0
    i1=0
    i0=0
    i3=0
    all_vel=[]
    for n in tqdm(range(frame_begin,frame_end+1)):
        i0=n
        i1=n+1
        i2=n+2
        i3=n+3

        '''_____________2値化画像読み込み______________'''
        a0=cv2_open(readfolder+"/th_%04d.bmp" %i0)#,frame_begin,frame_end)
        a1=cv2_open(readfolder+"/th_%04d.bmp" %i1)#,frame_begin,frame_end)
        a2=cv2_open(readfolder+"/th_%04d.bmp" %i2)#,frame_begin,frame_end)
        a3=cv2_open(readfolder+"/th_%04d.bmp" %i3)#,frame_begin,frame_end)

        '''_____________ラベリング＆重心算出______________'''
        nLabels_0, labeled_0, data_0, center_0 = cv2.connectedComponentsWithStats(a0)
        nLabels_1, labeled_1, data_1, center_1 = cv2.connectedComponentsWithStats(a1)
        nLabels_2, labeled_2, data_2, center_2 = cv2.connectedComponentsWithStats(a2)
        nLabels_3, labeled_3, data_3, center_3 = cv2.connectedComponentsWithStats(a3)

        '''_____________液滴追跡______________'''
        track_drop=track(center_0, center_1, center_2, center_3, rmax, rmin, re)

        '''_____________小オブジェクト速度除去______________'''
        vel_be=eli_ob(track_drop,data_0,diameter_min,diameter_max)

        '''_____________速度算出______________'''
        vel_af=velocity(track_drop, scales, time,vel_be)
        vels_af_only=vel_af_only(vel_af,velrange_min,velrange_max)
        all_vel.extend(vels_af_only)

    '''_____________全速度ヒストグラム化_____________'''
    vel_hist(date+ptv,all_vel,velrange_min,velrange_max)

    '''_____________平均速度保存_____________'''
    f=open(date+ptv+"/"+"condition.txt","w")
    f.write("平均速度:"+str(sum(all_vel)/len(all_vel))+"m/s\n")
    f.close()

    print('Finish')

'''<<<<<<<<<画像読み込み関数>>>>>>>>>>>>'''
def cv2_open(filename):
    img_ori = cv2.imread(filename)
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    return img_gray

'''<<<<<<<<<液滴追跡関数>>>>>>>>>>>>'''
def track(ps0,ps1,ps2,ps3,rmax,rmin,re):

    buffer = np.zeros((len(ps0),9),dtype=np.float64)
    count =0
    label_number = -1
    c=[]

    for p0 in ps0:

        x0=p0[0]
        y0=p0[1]
        label_number += 1
        element=None
        error0=re**2
        error_sum=(re**2)*2

        for p1 in ps1:
            x1=p1[0]
            y1=p1[1]

            distance0=math.sqrt((x1-x0)**2+(y1-y0)**2)

            if distance0>rmax or distance0<rmin:
                continue
            xd=x1+(x1-x0)
            yd=y1+(y1-y0)
            for p2 in ps2:
                x2 =p2[0]
                y2 =p2[1]

                distance1 =(x2-xd)**2+(y2-yd)**2

                if distance1>error0:
                    continue

                xd1=x2+(x2-x1)
                yd1=y2+(y2-y1)
                for p3 in ps3:
                    x3 =p3[0]
                    y3 =p3[1]

                    distance2 =(x3-xd1)**2+(y3-yd1)**2

                    if distance2>error0:
                        continue
                    error=distance1+distance2
                    if error>error_sum:
                        continue

                    error_sum=error

                    element=(p0,p1,p2,p3)
                    c.append(label_number)

        if element !=None:

            buffer[count][0]=element[0][0]
            buffer[count][1]=element[0][1]
            buffer[count][2]=element[1][0]
            buffer[count][3]=element[1][1]
            buffer[count][4]=element[2][0]
            buffer[count][5]=element[2][1]
            buffer[count][6]=element[3][0]
            buffer[count][7]=element[3][1]
            buffer[count][8]=c[-1]#検出粒子のラベリング番号
            count += 1
    return np.resize(buffer,(count,9))


'''<<<<<<<<<小オブジェクト速度除去関数>>>>>>>>>>>>'''
def eli_ob(v_d,size,diameter_min,diameter_max):
    vis=[]
    for i in range(len(v_d)):
        vi = int(v_d[i][8])
        aria=size[vi,4]
        if aria <= diameter_min:
            continue
        if aria >= diameter_max:
            continue
        vis.append(vi)
    return vis

'''<<<<<<<<<速度算出関数>>>>>>>>>>>>'''
def velocity(trajectories, scales, time, vis):
    size = len(vis)
    datasize = (size, 6)
    datatype = np.float64
    count=0
    result = np.empty(datasize, dtype=datatype)
    for i in range(len(trajectories)):
        if trajectories[i,8] in vis:
            t = trajectories[i]
            r = result[count]
            r[0] = t[0] #1枚目のx
            r[1] = t[1] #1枚目のy
            r[2] = (t[4] - t[0]) * scales[0] / 2 / time
            r[3] = (t[5] - t[1]) * scales[1] / 2 / time
            r[4] = 10*np.sqrt(r[2]**2+r[3]**2)/1000
            r[5] = t[8]
            count+=1
        else:
            continue
    return result

'''<<<<<<<<<速度抽出関数>>>>>>>>>>>>'''
def vel_af_only(v,velrange_min,velrange_max):
    vel=[]
    for i in range(len(v)):
        vel_1=v[i][4]
        vel.append(vel_1)
    return vel

'''<<<<<<<<<速度ヒストグラム化関数>>>>>>>>>>>>'''
def vel_hist(name,all_vel,velrange_min,velrange_max):
    fig= plt.figure(figsize=(8,6))
    plt.hist(all_vel, bins=41, range=(velrange_min,velrange_max), rwidth=0.5)
    plt.rcParams['font.family']='Arial'
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Count [-]')
    fig.savefig(name+'/all_velociy')
    plt.close()

'''<<<<<日時関数>>>>>>>>'''
def date():
    now = datetime.datetime.now()
    date=now.strftime('%y%m%d')
    return date

'''-----------メイン関数実行----------------'''
if __name__ == '__main__':
    date=date()
    main(date)
