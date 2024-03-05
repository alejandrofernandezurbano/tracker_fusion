import os
import open3d as o3d
import random
import glob
import tracker_fusion.YoloDetector as yd
import tracker_fusion.Lidar2Camera as l2c
import tracker_fusion.LidarUtils as lu
import tracker_fusion.Utils as ut
import tracker_fusion.FusionUtils as fu
import tracker_fusion.YoloUtils as yu					
import struct

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


def load_data(data_dir):
        image_files = sorted(glob.glob(data_dir+"images1/*.png"))
        point_files = sorted(glob.glob(data_dir+"pointspcd/*.pcd"))
        label_files = sorted(glob.glob(data_dir+"labels/*.txt"))
        calib_files = sorted(glob.glob(data_dir+"calibs/*.txt"))

        return image_files, point_files, label_files, calib_files
        
def main(_argv):

    index = 1
    data_dir = "/media/alejandro-fernandez/6465-3263/Tesis/MyDataset/"
    imgs, pts, labels, calibs = load_data(data_dir)
    weights = data_dir + "model//yolov4//yolov4.weights"
    config = data_dir + "model//yolov4//yolov4.cfg"
    names = data_dir + "model//yolov4//coco.names"
    out_dir = os.path.join(data_dir, "output//images")
    
    detector = yd.Detector(0.4)
    detector.load_model(weights, config, names)

#######################################################################################################################################################################################
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    
    input_size = FLAGS.size
    video_path = FLAGS.video

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print("          ")
        print('Frame #: ', frame_num)
        start_time = time.time()               ##################################################################################################################################################################################################       
        detections, yolo_detections = detector.detect(frame, draw_bboxes=True, display_labels=True)
        
        bboxess = [subarreglo[1] for subarreglo in detections]
        scoress = np.array([subarreglo[2] for subarreglo in detections], dtype=np.float64)
        scoress_array = np.array([subarreglo[0] for subarreglo in scoress], dtype=np.float64)

        class_namess = [subarreglo[3] for subarreglo in detections]
        bboxess = np.array(bboxess, dtype=np.float64)
        #scoress = np.array(scoress)
        class_namess= np.array(class_namess)
        features2 = encoder(frame, bboxess)
        print("          ")
        print("bboxes : ",      bboxess)
        print("scores : ",     scoress_array)
        print("class_names : ", class_namess)
        print("          ")   
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxess, scoress_array, class_namess, features2)] ##################################################################################################################################################################################################    
        #detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
