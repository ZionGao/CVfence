
import cv2
import numpy as np
import math
import time

poly = []

def set_area(img,area):
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            print(xy)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            poly.append([float(x), float(y)])
            area.append([float(x), float(y)])
            print(len(poly))
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)

    while (len(poly)<4 ):
        try:
            cv2.waitKey(100)
        except Exception:
            cv2.destroyWindow("image")
            break
    cv2.destroyWindow("image")




def winding_number(point):
    poly.append(poly[0])
    px = point[0]
    py = point[1]
    sum_of_p = 0
    length = len(poly) - 1
    # length = len(poly)

    for index in range(0, length):
        sx = poly[index][0]
        sy = poly[index][1]
        tx = poly[index + 1][0]
        ty = poly[index + 1][1]

        # The points coincide with the vertices of a polygon or are on the edges of a polygon
        if ((sx - px) * (px - tx) >= 0 and (sy - py) * (py - ty) >= 0 and (px - sx) * (ty - sy) == (py - sy) * (
                tx - sx)):
            return "on"
        # The Angle between a point and an adjacent vertex
        angle = math.atan2(sy - py, sx - px) - math.atan2(ty - py, tx - px)

        # Make sure the Angle is within the range（-π ~ π）
        if angle >= math.pi:
            angle = angle - math.pi * 2
        elif angle <= -math.pi:
            angle = angle + math.pi * 2
        sum_of_p += angle

        # Calculate the number of turns and judge the geometric relationship between points and polygons
    result = 'out' if int(sum_of_p / math.pi) == 0 else 'in'
    return result


def get_poly():
    pts = np.array(poly, np.int32)
    # 顶点个数：4，矩阵变成4*1*2维,第一个参数为-1, 表明长度是根据后面的维度计算的。
    pts = pts.reshape((-1, 1, 2))
    return pts


def person_in_area(q, flg=False):
    '''
    判断进入区域的动作
    :param q: 人物的位置坐标的队列
    :param flg: 开始的状态，默认在区域内
    :return:
    '''
    while True:
        if not q:
            return "non"
        box1 = q.popleft()
        if winding_number(box1) == "out":
            flg = True
            continue
        elif winding_number(box1) == "in" and flg:
            return "yes"


def get_out_or_in(q, flg=False):
    """
    :param q: 人轨迹坐标形成的队列，坐标为（x,y,w,h）
    :param flg: 一个开关，管理最开始的状态，默认False表明开始状态为在区域内，True表示在区域外
    判断人是否属于撞线：
        ①out-->in-->左侧线相交
        ②out-->in-->右侧线相交
        ③out-->in-->上侧线相交
        ④out-->in持续5秒
        ⑤out-->in-->消失5秒
    :return: yes/non，其中yes指触发报警，non指其他情况不报警
    """
    print("进入程序")
    flg_1 = False
    while True:
        if not q:
            print("空")
            return "non"
        box1 = q.popleft()
        if winding_number(box1) == "out":
            flg = True
            print("it is out")
        elif winding_number(box1) == "in" and flg:
            start_time = time.time()
            flg_1 = True
            print("it is in")
            break
        else:
            print("other")
    while flg_1:
        now_time = time.time()
        # print(now_time, start_time)
        if int(now_time) - int(start_time) > 1:
            print("时间报警")
            return "yes"
        if q:
            box = q.popleft()
            if winding_number(box) == "out":
                flg_1 = False
                print("又重新out了")
                get_out_or_in(q, True)
            # 离开区域行为的坐标设置及时间设置，可以修改数值
            if box[1] <= 120 and int(now_time) - int(start_time) > 1:
                print("位置报警")
                return "yes"
    return "non"


def cross_point(line1, line2):  # Compute the intersection function
    """
    eg:
    point1 = [80,37,501,366]
    point2 = [179,16,110,504]
    point_is_exist, [x, y] = cross_point(point1, point2)
    print(point_is_exist,[x,y])
    """
    point_is_exist = False
    x = 0
    y = 0
    x1 = line1[0]  # Get the four point coordinate
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True
    return point_is_exist, [x, y]
