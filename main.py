#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
import argparse

from PIL import Image
from yolo import YOLO
from collections import deque

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from keras import backend
import crash_area as clt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# import crash_area

backend.clear_session()
warnings.filterwarnings('ignore')

arg = argparse.ArgumentParser()
arg.add_argument("--input", default="test.mp4", help="path to input video", )

args = vars(arg.parse_args())

dqs = [deque() for _ in range(9999)]

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture(args["input"])
    # video_capture = cv2.VideoCapture(0)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    queue_dict = {}
    id_cnt_dict = {}


    cnt = 0

    ret, frame = video_capture.read()
    clt.set_area(frame)


    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, ret_clss = yolo.detect_image(image)
        features = encoder(frame, boxs)  # The image of each frame is coded to match the box
        # score to 1.0 here).  Each detection box and feature is encapsulated as an object
        detections = [Detection(bbox, 1.0, feature, ret_cls) for bbox, feature, ret_cls in
                      zip(boxs, features, ret_clss)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # nms_max_overlap的值为1，没有进行nms操作
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        index_ids = []

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        cur = []

        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            bcls = track.track_cls
            t_id = track.track_id

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
                dqs[track.track_id].append(center)
                if len(dqs[track.track_id])>30:
                    dqs[track.track_id].popleft()
                thickness = 4
                # draw the center
                cv2.circle(frame, center, 1, color, thickness)
                # Draw the trajectory of the character's movement
                for j in range(1, len(dqs[track.track_id])):
                    if dqs[track.track_id][j - 1] is None or dqs[track.track_id][j] is None:
                        continue
                    # thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    thickness = 3
                    cv2.line(frame, (dqs[track.track_id][j - 1]), (dqs[track.track_id][j]), (color), thickness)
                queue_dict[track.track_id] = dqs[track.track_id]

                if clt.winding_number(center) == 'in':
                    if clt.person_in_area(queue_dict[t_id]) == "yes" and t_id not in id_cnt_dict.keys():
                        image.save("output/{}.jpg".format(cnt))
                        id_cnt_dict[t_id] = 1
                        cnt += 1
                        queue_dict[t_id].clear()

        # Paint the door
        pts1 = clt.get_poly()
        cv2.polylines(frame, [pts1], True, (0, 255, 0), thickness=3)

        cv2.putText(frame, "FPS: %f" % fps, (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.putText(frame, "cnt: %d" % cnt, (int(20), int(80)), 0, 5e-3 * 200, (255, 0, 0), 3)
        # Can modify the size of the display window
        cv2.imshow('object-tracking', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
