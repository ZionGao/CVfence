import cv2
import numpy as np
from collections import deque


def initArea( image):
    '''
    点选区域左上角 右下角
    :param image: hape w*h*3
    :return:
    '''

    boxes = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            print(xy)
            cv2.circle(image, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            boxes.append([int(x), int(y)])
            cv2.imshow("image", image)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", image)



    while (len(boxes) < 2):
        try:
            cv2.waitKey(100)
        except Exception:
            cv2.destroyWindow("image")
            break


    return boxes


def isChange(p1, p2, gauss_image):
    x1, y1 = p1
    x2, y2 = p2
    npim = np.zeros((y2 - y1, x2 - x1), dtype=np.int)
    npim[:] = gauss_image[y1:y2, x1:x2]
    wt = len(npim[npim > 0])
    bl = len(npim[npim == 0])
    c = wt / (wt + bl)
    return c


# cap = cv2.VideoCapture("/Users/gao/PycharmProjects/CVfence/test.mp4")
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

ret, frame = cap.read()
box = initArea(frame)
q = deque()

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        tempframe = frame
        if (len(q)<10):
            q.append(tempframe)
            print(len(q))
        if (len(q) == 10):
            pre = q[0]
            cur = q[9]
            q.popleft()
            q.append(tempframe)
            previousframe = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
            currentframe = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            currentframe = cv2.absdiff(currentframe, previousframe)
            median = cv2.medianBlur(currentframe, 3)

            ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)
            gauss_image = cv2.GaussianBlur(threshold_frame, (3, 3), 0)

            c1 = isChange(box[0],box[1],gauss_image)




            print('*'*30)

            print("区域1变动率{}".format(c1))

            if c1> 0.1:
                print('区域人员入侵')

            cv2.rectangle(frame, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), (255, 0, 0), 2)




            # # Display the resulting frame
            cv2.imshow('pic', frame)
            #
            # # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()