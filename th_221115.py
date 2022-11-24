import cv2
import os

"""------変更箇所--------"""
last = 500
maisu = 501//100
step = 1
readfile = 'result2'
fps = 10000
writefile = 'triming3'
"""------main--------"""
if fps == 20000:
    ru ,rd , lu ,ld = 90, 190, 100, 1100
if fps == 10000:
    ru ,rd , lu ,ld = 200, 300, 100, 1100

"""フォルダ作成"""
os.makedirs(writefile + '/ori',exist_ok=True)
os.makedirs(writefile + '/th',exist_ok=True)
os.makedirs(writefile + '/kukei',exist_ok=True)
os.makedirs(writefile,exist_ok=True)

dic={}
otsu_all = 0
ret_otsu_all = 0

"""画像読み込み"""
for i in range(0, last, step):
    img = cv2.imread(readfile+'/sample_video_img_%04d.bmp' % i)
    img1 = img[ru : rd, lu: ld]   
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(writefile + "/ori/triming_%04d.bmp" %i, gray)

    """画像フィルタ（ガウシアンフィルタ）"""
    img_gafilter = cv2.GaussianBlur(gray,     # 入力画像
                               (3,3),    # カーネルの縦幅・横幅
                                0        # 横方向の標準偏差（0を指定すると、カーネルサイズから自動計算）
                            )

    """動的二値化"""
    th2_ga = cv2.adaptiveThreshold(img_gafilter,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,7,-5)
    #cv2.imwrite("triming/th/triming_%04d.bmp" %i, img_gafilter)
    cv2.imwrite(writefile + "/th/th_%04d.bmp" %i, th2_ga)


    nLabels_1, labeled_1, data_1, center_1 = cv2.connectedComponentsWithStats(th2_ga)
    aria_1 = 0
    for k in range(1,nLabels_1):
            x, y, w, h, size = data_1[k,0],data_1[k,1],data_1[k,2],data_1[k,3],data_1[k,4]
            otsu_rec = cv2.rectangle(img1,(x-1,y-1),(x+w,y+h),(0,255,0),1)
            aria_1 += size
    cv2.imwrite(writefile + "/kukei/kukei_%04d.bmp" %i, otsu_rec)

    """
    void_1 = aria_1 / data_1[0][4] * 100
    otsu_all += void_1
    ret_otsu_all += ret_otsu
    f = open('triming/otsu/aria.txt','a')
    f.write("ボイド率(大津{}枚目)(閾値={}):".format(i,ret_otsu)+str(void_1)+"%\n")
    f.close()

    for j in range(20,201,10):
        os.makedirs('triming/{}'.format(j),exist_ok=True)
        os.makedirs('triming/{}/ori'.format(j),exist_ok=True)
        os.makedirs('triming/{}/th'.format(j),exist_ok=True)
        ret_img, th_img = cv2.threshold(gray, j, 255, cv2.THRESH_BINARY)
        cv2.imwrite("triming/{}/th/triming_%04d.bmp".format(j) %i, th_img)

        nLabels_0, labeled_0, data, center_0 = cv2.connectedComponentsWithStats(th_img)

        aria = 0
        for k in range(1,nLabels_0):
            x, y, w, h, size = data[k,0],data[k,1],data[k,2],data[k,3],data[k,4]
            img_rec = cv2.rectangle(img1,(x-1,y-1),(x+w,y+h),(0,255,0),1)
            aria += size

        cv2.imwrite("triming/{}/ori/triming_%04d.bmp".format(j) %i, img_rec)
        void = aria / data[0][4] * 100
        
        if i == 0:
            dic[j] = void
        else:
            dic[j] = void + dic[j]
        # print(dic)
        f = open('triming/{}/aria.txt'.format(j),'a')
        f.write("ボイド率({}枚目):".format(i)+str(void)+"%\n")
        f.close()
    print(i)
    if i == last-1:
        print(i)
        for l in range(20,201,10):
            f = open('triming/all_aria.txt','a')
            f.write("ボイド率平均(閾値={}):".format(l)+str((dic[l])/(maisu))+"%\n")
            f.close()
f = open('triming/all_aria.txt','a')
f.write("大津法ボイド率平均(平均閾値={}):".format(ret_otsu_all/maisu)+str((otsu_all)/(maisu))+"%\n")
f.close()

"""
