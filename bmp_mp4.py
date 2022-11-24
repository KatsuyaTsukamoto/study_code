import sys
import cv2

# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
video = cv2.VideoWriter('video_sabun_1.0.mp4',fourcc, 10, (1000, 100))

if not video.isOpened():
    print("can't be opened")
    sys.exit()

for i in range(0, 90+1):
    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv2.imread('holdtime_sabun_1/image_kukei-%04d.bmp' % i)

    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)
    print(i)

video.release()
print('written')