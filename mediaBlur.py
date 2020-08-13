import cv2
import numpy as np
import queue


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




cap = cv2.VideoCapture("/Users/gao/PycharmProjects/CVfence/test.mp4")
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
frameNum = 0
# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    global box
    frameNum += 1
    if ret == True:
        tempframe = frame
        if (frameNum == 1):
            box = initArea(tempframe)
            previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            print(111)
        if (frameNum % 10 == 0):
        # if (frameNum >= 2):

            currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
            currentframe = cv2.absdiff(currentframe, previousframe)
            median = cv2.medianBlur(currentframe, 3)

            ret, threshold_frame = cv2.threshold(currentframe, 20, 255, cv2.THRESH_BINARY)
            gauss_image = cv2.GaussianBlur(threshold_frame, (3, 3), 0)


            h, w = gauss_image.shape

            def isChange(p1,p2):
                x1, y1 = p1
                x2, y2 = p2
                npim = np.zeros((y2 - y1, x2 - x1), dtype=np.int)
                npim[:] = gauss_image[y1:y2, x1:x2]
                wt = len(npim[npim > 0])
                bl = len(npim[npim == 0])
                c = wt/(wt+bl)
                return c

            c1 = isChange(box[0],box[1])
            # c2 = isChange(box[2], box[3])
            # c3 = isChange(box[4], box[5])



            print('*'*30)

            print("区域1变动率{}".format(c1))
            # print("区域2变动率{}".format(c2))
            # print("区域3变动率{}".format(c3))
            print('*' * 30)




            # Display the resulting frame
            cv2.imshow('原图', frame)
            # cv2.imshow('Frame', currentframe)
            # cv2.imshow('median', median)
            cv2.imshow('gauss', gauss_image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()