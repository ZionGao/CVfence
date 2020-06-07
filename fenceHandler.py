#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import warnings
import numpy as np
import math
import cv2
from PIL import Image
from yolo import YOLO
from collections import deque
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
# from keras import backend
from tensorflow.keras import backend
import tensorflow as tf
import os
import time


os.environ['KMP_DUPLICATE_LIB_OK']='True'
config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.5)
config.gpu_options.allow_growth = True


backend.clear_session()
warnings.filterwarnings('ignore')
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

class Fence():

    def __init__(self):
        self.yolo = YOLO()
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        self.encoder = gdet.create_box_encoder('model_data/mars-small128.pb', batch_size=1)
        self.tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget))
        self.dqs = [deque() for _ in range(9999)]
        self.poly = []
        self.id_cnt_dict = {}
        self.queue_dict = {}

    def initArea(self,image):
        '''
        初始化敏感区域 输入图像点选保存坐标
        :param image: hape w*h*3
        :return:
        '''
        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                xy = "%d,%d" % (x, y)
                print(xy)
                cv2.circle(image, (x, y), 1, (255, 0, 0), thickness=-1)
                cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 0), thickness=1)
                self.poly.append([float(x), float(y)])
                cv2.imshow("image", image)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        cv2.imshow("image", image)

        while (len(self.poly) < 4):
            try:
                cv2.waitKey(100)
            except Exception:
                cv2.destroyWindow("image")
                break
        cv2.destroyWindow("image")
        return self.poly


    def initPoly(self,poly):
        '''
        初始化敏感区域
        :param poly: 输入区域坐标顶点[[float(x), float(y)]...]
        :return:
        '''
        self.poly = poly

    def detect(self,image):
        '''
        只用检测模型获取bbox，输出不包含目标id
        :param image: shape w*h*3
        :return: bboxes[[min x, min y, max x, max y]...]
        '''
        img = Image.fromarray(image[..., ::-1])  # bgr to rgb
        boxs, ret_clss = self.yolo.detect_image(img)

        boxs = [boxs[i] for i in range(0,len(ret_clss)) if ret_clss[i] == ['person']]
        ret_clss = [['person']*len(boxs)]

        features = self.encoder(image, boxs)  # The image of each frame is coded to match the box
        # score to 1.0 here).  Each detection box and feature is encapsulated as an object
        detections = [Detection(bbox, 1.0, feature, ret_cls) for bbox, feature, ret_cls in
                      zip(boxs, features, ret_clss)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        bboxes = []

        for det in detections:
            bbox = det.to_tlbr()
            bboxes.append(bbox)

        return bboxes, ret_clss, detections

    def trackByDetect(self,image):
        '''
        检测跟踪获取bbox
        :param image: shape w*h*3
        :return: b   [[min x, min y, max x, max y]...]
                 t   [id1,id2,id3...]
        '''

        bbox, ret_clss,detections = self.detect(image)

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        b = []
        c = []
        t = []

        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            b.append(track.to_tlbr())
            c.append(track.track_cls)
            t.append(track.track_id)

        return  b,c,t


    def entanceAlert(self,bboxes):
        '''
        判断bbox中心是否在敏感区域中
        :param bboxes: [[min x, min y, max x, max y]...]
        :return: isIn :[True,False,True....]
        '''
        assert len(self.poly) == 4 , '未初始化四边形铭感区域'
        isIn = [ self.winding_number(int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2)) == 'in' for bbox in bboxes  ]
        return isIn


    def winding_number(self,point):
        '''
        区域坐标与中心坐标关系判断
        :param point: [x,y]
        :return: "on" 边界线上
                 "in" 区域内
                 "out" 区域外
        '''
        self.poly.append(self.poly[0])
        px = point[0]
        py = point[1]
        sum_of_p = 0
        length = len(self.poly) - 1
        # length = len(poly)

        for index in range(0, length):
            sx = self.poly[index][0]
            sy = self.poly[index][1]
            tx = self.poly[index + 1][0]
            ty = self.poly[index + 1][1]

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

    def get_poly(self):
        pts = np.array(self.poly, np.int32)
        # 顶点个数：4，矩阵变成4*1*2维,第一个参数为-1, 表明长度是根据后面的维度计算的。
        pts = pts.reshape((-1, 1, 2))
        return pts

    def person_in_area(self,q, flg=False):
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
            if self.winding_number(box1) == "out":
                flg = True
                continue
            elif self.winding_number(box1) == "in" and flg:
                return "yes"

def main():


    video_capture = cv2.VideoCapture("test.mp4")
    fps = 0.0
    cnt = 0


    ret, frame = video_capture.read()

    # 1 初始化电子围栏对象
    f = Fence()
    f.initArea(frame)
    tracker = f.tracker

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        #  2 输入图像获取跟踪信息
        b,c,t = f.trackByDetect(frame)

        tracks = zip(b,c,t)

        i = int(0)
        index_ids = []


        cur = []

        for track in tracks:

            bbox = track[0]
            bcls = track[1]
            t_id = track[2]

            index_ids.append(int(t_id))
            color = [int(c) for c in COLORS[index_ids[i] % len(COLORS)]]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

            str1 = bcls[0]
            cv2.putText(frame, str1, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
            i += 1


            if str(bcls[0]) == "person":

                cur.append(t_id)
                # Calculate the center coordinates of bbox
                center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
                # track_id[center]
                f.dqs[t_id].append(center)
                if len(f.dqs[t_id])>30:
                    f.dqs[t_id].popleft()
                thickness = 4
                # draw the center
                cv2.circle(frame, center, 1, color, thickness)
                # Draw the trajectory of the character's movement
                for j in range(1, len(f.dqs[t_id])):
                    if f.dqs[t_id][j - 1] is None or f.dqs[t_id][j] is None:
                        continue
                    # thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    thickness = 3
                    cv2.line(frame, (f.dqs[t_id][j - 1]), (f.dqs[t_id][j]), (color), thickness)
                f.queue_dict[t_id] = f.dqs[t_id]

                if f.winding_number(center) == 'in':
                    if f.person_in_area(f.queue_dict[t_id]) == "yes" and t_id not in f.id_cnt_dict.keys():
                        f.id_cnt_dict[t_id] = 1
                        cnt += 1
                        f.queue_dict[t_id].clear()

        # Paint the door
        pts1 = f.get_poly()
        cv2.polylines(frame, [pts1], True, (0, 255, 0), thickness=3)

        cv2.putText(frame, "FPS: %f" % fps, (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.putText(frame, "cnt: %d" % cnt, (int(20), int(80)), 0, 5e-3 * 200, (255, 0, 0), 3)
        # Can modify the size of the display window
        cv2.imshow('object-tracking', frame)



        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
