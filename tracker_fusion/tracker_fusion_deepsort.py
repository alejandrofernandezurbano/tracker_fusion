import os
import open3d as o3d
#print(o3d.__version__)

import random
import glob
import tracker_fusion.YoloDetector as yd
import tracker_fusion.Lidar2Camera as l2c
import tracker_fusion.LidarUtils as lu
import tracker_fusion.Utils as ut
import tracker_fusion.FusionUtils as fu
import tracker_fusion.YoloUtils as yu					
import struct
#
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2,PointField
from std_msgs.msg import String
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from rclpy.qos import QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import cv2
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import tracker_fusion.utils as utils
from tracker_fusion.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from tracker_fusion.config import cfg
#from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
import tracker_fusion.preprocessing as pp
import tracker_fusion.nn_matching as nn
#from deep_sort.detection import Detection
from tracker_fusion.detection import Detection
from tracker_fusion.tracker import Tracker as Track
import tracker_fusion.generate_detections as gdet
flags.DEFINE_string('output', './src/tracker_fusion/tracker_fusion/outputs/testros.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('fusion_subscriber')
        qos_profile = rclpy.qos.qos_profile_sensor_data
        self.subscription_img = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos_profile)

        self.subscription_lidar = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.lidar_callback,
            qos_profile)
        self.publisher_tracking = self.create_publisher(Image, 'tracked_image', qos_profile)
        #self.publisher = self.create_publisher(String, 'image_save_signal', 10)
        self.data_dir = "/home/chimuelo/fusion_ws/src/tracker_fusion/tracker_fusion/Code/Data/"
        self.image_count = 0
        self.intrinsics = None
        self.distortions = None
        self.msgvelo = None
        self.frame_num = 0
        self.bridge = CvBridge()
        # by default VideoCapture returns float instead of int
        self.width = 640         
        self.height = 480
        #fps = int(vid.get(cv2.CAP_PROP_FPS))
        self.codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.out = cv2.VideoWriter('./src/tracker_fusion/tracker_fusion/outputs/test_ros.avi', self.codec,2.2,(self.width, self.height)) 
        
        self.weights = self.data_dir + "model//yolov4//yolov4.weights"
        self.config = self.data_dir + "model//yolov4//yolov4.cfg"
        self.names = self.data_dir + "model//yolov4//coco.names" 
        self.detector = yd.Detector(0.4)
        self.detector.load_model(self.weights, self.config, self.names)
        print("Done")      
    def load_data(self,data_dir):
                
        label_files = sorted(glob.glob(data_dir+"label/*.txt"))
        calib_files = sorted(glob.glob(data_dir+"calib/*.txt"))

        return label_files, calib_files
    def image_callback(self, msg):
    
        labels, calibs = self.load_data(self.data_dir)
           
        out_dir = os.path.join(self.data_dir, "output//images") 

                       
        #cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #cv2.imshow('Image', cv_image)

        
        if self.msgvelo is not None:
#########################################################################################################################################################################
            # Definition of the parameters
            max_cosine_distance = 0.3
            nn_budget = None
            nms_max_overlap = 1.0
        
            # initialize deep sort
            model_filename = '/home/chimuelo/fusion_ws/src/tracker_fusion/tracker_fusion/model_data/mars-small128.pb'
            encoder = gdet.create_box_encoder(model_filename, batch_size=1)
            # calculate cosine distance metric
            metric = nn.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            # initialize tracker
            tracker = Track(metric)
#########################################################################################################################################################################                          
 
            self.frame_num +=1
            print("          ")
            print('Frame #: ', self.frame_num)
            start_time = time.time()
            # load the image
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # create LiDAR2Camera object
            lidar2cam = l2c.LiDAR2Camera(calibs[0]) 
            # 1 - Run 2D object detection on image
            detections, yolo_detections = self.detector.detect(image, draw_bboxes=True, display_labels=True)
            # Extract point cloud data
            gen = pc2.read_points(self.msgvelo, field_names=("x", "y", "z"), skip_nans=True)
            # Convert point cloud data to NumPy array         
            point_cloud = np.array(list(gen))
            # load lidar points and project them inside 2d detection
            #point_cloud = np.asarray(o3d.io.read_point_cloud(pts[frame_num-1]).points) # original 11
            pts_3D, pts_2D = lu.get_lidar_on_image(lidar2cam, point_cloud, (image.shape[1], image.shape[0]))
            lidar_pts_img, _ = fu.lidar_camera_fusion(pts_3D, pts_2D, detections, image)
            # Build a 2D Object
            list_of_2d_objects = ut.fill_2D_obstacles(detections)
            # Build a 3D Object (from labels)
            list_of_3d_objects = ut.read_label(labels[0]) # original 5 #### DET3D
            # Get the LiDAR Boxes in the Image in 2D and 3D
            lidar_2d, lidar_3d = lu.get_image_with_bboxes(lidar2cam, lidar_pts_img, list_of_3d_objects)
            # Associate the LiDAR boxes and the Camera Boxes
            lidar_boxes = [obs.bbox2d for obs in list_of_3d_objects]  # Simply get the boxes
            pred_bboxes = [detection[1] for detection in detections]
            camera_boxes = [np.array([box[0], box[1], box[0] + box[2], box[1]+box[3]]) for box in pred_bboxes]
            
            matches, unmatched_lidar_boxes, unmatched_camera_boxes = fu.associate(lidar_boxes, camera_boxes)               #ERRRRORRRRR
            #print("matches: ",matches);print("unmatched_lidar_boxes: ",unmatched_lidar_boxes);print("unmatched_camera_boxes: ",unmatched_camera_boxes)
            # Build a Fused Object
            final_image, list_of_fused_objects = fu.build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, lidar_2d)
            final_image2, _ = fu.build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, lidar_2d)
            #print("list_of_fused_objects : ", list_of_fused_objects)
            # draw yolo detections on top to fused results
            final_image = yu.draw_yolo_detections(final_image, detections,classes_to_draw=("person", "car"))
        
            bboxessfu = [subarreglo[0] for subarreglo in list_of_fused_objects]
            bboxessfu2 = np.array(bboxessfu, dtype=np.float64)
            bboxessfu = np.array(bboxessfu, dtype=np.float64)
            bboxessfu = np.array([[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in bboxessfu], dtype=np.float64)       
            scoressfu = np.array([subarreglo[2] for subarreglo in list_of_fused_objects], dtype=np.float64)
            class_namessfu = [subarreglo[1] for subarreglo in list_of_fused_objects]
            class_namessfu = np.array(class_namessfu)
            features3 = encoder(image, bboxessfu)
            for bbox in bboxessfu2:
                bbox = bbox.astype(int)  # Asegúrate de que las coordenadas sean enteras
                cv2.rectangle(image2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(out_dir,f"lidar_2dAGX.png"), yolo_detections)    
            cv2.imwrite(os.path.join(out_dir,f"lidar_3dAGX.png"), final_image2)
            cv2.imwrite(os.path.join(out_dir,f"opti_malAGX{self.frame_num-1}.png"), image2)        

#########################################################################################################################################################################
          
            bboxess = [subarreglo[1] for subarreglo in detections]
            scoress = np.array([subarreglo[2] for subarreglo in detections], dtype=np.float64)
            scoress_array = np.array([subarreglo[0] for subarreglo in scoress], dtype=np.float64)
            class_namess = [subarreglo[3] for subarreglo in detections]
            bboxess = np.array(bboxess, dtype=np.float64)
            class_namess= np.array(class_namess)
            features2 = encoder(image, bboxess)
        
            #print("          ")
            #print("bboxes : ",      bboxess)
            #print("scores : ",     scoress_array)
            #print("class_names : ", class_namess)
            #print("          ")   
        
            #YOLOV4
            #detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxess, scoress_array, class_namess, features2)]
            #FUSION
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxessfu, scoressfu, class_namessfu, features3)]
        
            #cv2.imshow("final_image", final_image)
            #cv2.waitKey(0) 
#########################################################################################################################################################################  
            #detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = pp.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                #if not track.is_confirmed() or track.time_since_update > 1:
                    #print("No Funciona")
                    #continue 
                #print("Funciona")
                bbox = track.to_tlbr()
                class_name = track.get_class()
            
            # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(image, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        

            # if enable info flag then print details about each track
                #if True:
                    #print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]),
                    #int(bbox[2]), int(bbox[3]))))
            cv2.imwrite(os.path.join(out_dir,f"track_fusin{(self.frame_num-1)*2}.png"), image)      #PARES FUSION
            #cv2.imwrite(os.path.join(out_dir,f"track_fusin{frame_num*2-1}.png"), image)        #IMPARES YOLOV4
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(image)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            #if not FLAGS.dont_show:
            #cv2.imshow("Output Video", result)
            #cv2.waitKey(0)

          
            #Correr Nodo Publicador
            tracked_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            self.publisher_tracking.publish(tracked_image_msg)
            
            #Guardar en video.
            self.out.write(result)
            
        cv2.destroyAllWindows()
            #cv2.namedWindow("lidar_pts_img", cv2.WINDOW_NORMAL)  
            #cv2.imshow("lidar_pts_img", lidar_pts_img)
            #cv2.namedWindow("YOLO_img", cv2.WINDOW_NORMAL)  
            #cv2.imshow("YOLO_img", yolo_detections)
            #key = cv2.waitKey(1)
            
              
    def info_callback(self, msg):
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        self.intrinsics = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])

        k1, k2, p1, p2, k3 = msg.D
        self.distortions = np.array([k1, k2, p1, p2, k3])

    def lidar_callback(self, msg):
        # Process Lidar data here
        # For example, you can convert PointCloud2 message to a numpy array
        #self.point_cloud_data = np.asarray(msg.data, dtype=np.float32).reshape(-1, 3)
        self.msgvelo = msg
    
def main(args=None):
    rclpy.init(args=args)
    fusion_subscriber = ImageSubscriber()
    rclpy.spin(fusion_subscriber)
    fusion_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

