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
import tracker_fusion.Fusion as fus
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
###{DET3D########################################################################################
import argparse
import glob
from pathlib import Path
import open3d
import tracker_fusion.open3d_vis_utils as V
import torch
import tracker_fusion.demo as det3d
###DET3D}########################################################################################
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import cv2


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
             
        self.weights = self.data_dir + "model//yolov4//yolov4.weights"
        self.config = self.data_dir + "model//yolov4//yolov4.cfg"
        self.names = self.data_dir + "model//yolov4//coco.names" 
        self.detector = yd.Detector(0.4)
        self.detector.load_model(self.weights, self.config, self.names)
        ###################################################################
        #self.demo_dataset, self.model3d = det3d.model3d()
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
            points = []
            point_step = self.msgvelo.point_step
            for i in range(0, len(self.msgvelo.data), point_step):
                data = struct.unpack_from('ffff', self.msgvelo.data, offset= i)
                x, y, z, intensity = data
                points.append([x, y, z,0.0])         
            points = np.array(points)
            pred_dicts = det3d.model3d(points)
#########################################################################################################################################################################                          
            self.frame_num +=1
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
#########################################################################################################################################################################
            # Build a 3D Object (from labels)
            list_of_3d_objects = ut.read_label(pred_dicts) # original 5   
            #list_of_3d_objects = [fus.Object3D(pred_boxes.cpu().numpy(),pred_scores.cpu().numpy(),pred_labels.cpu().numpy())]                     #### DET3D
#########################################################################################################################################################################
            # Get the LiDAR Boxes in the Image in 2D and 3D
            lidar_2d, lidar_3d = lu.get_image_with_bboxes(lidar2cam, lidar_pts_img, list_of_3d_objects)
            # Associate the LiDAR boxes and the Camera Boxes
            lidar_boxes = [obs.bbox2d for obs in list_of_3d_objects]  # Simply get the boxes
            pred_bboxes = [detection[1] for detection in detections]
            camera_boxes = [np.array([box[0], box[1], box[0] + box[2], box[1]+box[3]]) for box in pred_bboxes]
            
            matches, unmatched_lidar_boxes, unmatched_camera_boxes = fu.associate(lidar_boxes, camera_boxes)               #ERRRRORRRRR
            print("matches: ",matches);
            print("unmatched_lidar_boxes: ",unmatched_lidar_boxes);print("unmatched_camera_boxes: ",unmatched_camera_boxes)
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

            for bbox in bboxessfu2:
                bbox = bbox.astype(int)  # Asegúrate de que las coordenadas sean enteras
                cv2.rectangle(image2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(out_dir,f"lidar_2dAGX.png"), yolo_detections)    
            cv2.imwrite(os.path.join(out_dir,f"lidar_3dAGX.png"), final_image2)
            cv2.imwrite(os.path.join(out_dir,f"opti_malAGX{self.frame_num-1}.png"), image2)     
               
            result = np.asarray(image2)
            result = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        
            #Correr Nodo Publicador
            tracked_image_msg = self.bridge.cv2_to_imgmsg(result, encoding="bgr8")
            self.publisher_tracking.publish(tracked_image_msg)

            
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

