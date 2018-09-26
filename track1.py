import numpy as np
import cv2
import pickle
toBePro = open('position.pkl', 'rb')
countNum = open('count.pkl', 'rb')

cap = cv2.VideoCapture('../192.168.4.50_2018-08-15_16-20-00_2018-08-15_16-38-49.mp4')

fps = 10  # 保存视频的FPS，可以适当调整
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('saveVideo.avi', fourcc, fps, (480, 360))  # 最后一个是保存图片的尺寸for imgname in imgs:




while (True):
    # capture frame-by-frame
    ret, frame = cap.read()
    a = pickle.load(toBePro)
    countTotal = pickle.load(countNum)

    # our operation on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体

    for item in a:
        cv2.rectangle(frame, (int(item[0]), int(item[1])), (int(item[0]) + 13, int(item[1])+13), (0, 244, 0), 4)
        # print(item[0], item[1])

    frame = cv2.putText(frame, str(countTotal[0][0]), (879, 425), font, 1.2, (0, 255, 0), 2)  # 添加文字（图片，要添加的字符，位置， 字体，字体大小，颜色，字体粗细
    frame = cv2.putText(frame, str(countTotal[1][0]), (842, 446), font, 1.2, (0, 255, 0), 2)  # 添加文字（图片，要添加的字符，位置， 字体，字体大小，颜色，字体粗细
    frame = cv2.putText(frame, str(countTotal[2][0]), (811, 465), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[3][0]), (774, 486), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[4][0]), (724, 517), font, 1.2, (0, 255, 0), 2)

    frame = cv2.putText(frame, str(countTotal[5][0]), (1359, 379), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[6][0]), (1328, 362), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[7][0]), (1302, 343), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[8][0]), (1276, 327), font, 1.2, (0, 255, 0), 2)

    frame = cv2.putText(frame, str(countTotal[9][0]), (1366, 657), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[10][0]), (1392, 628), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[11][0]), (1421, 607), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[12][0]), (1446, 589), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[13][0]), (1486, 563), font, 1.2, (0, 255, 0), 2)

    frame = cv2.putText(frame, str(countTotal[14][0]), (807, 724), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[15][0]), (839, 744), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[16][0]), (866, 772), font, 1.2, (0, 255, 0), 2)
    frame = cv2.putText(frame, str(countTotal[17][0]), (1, 35), font, 1, (0, 255, 0), 2)
    # display the resulting frame
    # cv2.imshow('frame11111', frame)
    videoWriter.write(frame)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break
# when everything done , release the capture

videoWriter.release()

cap.release()
cv2.destroyAllWindows()




