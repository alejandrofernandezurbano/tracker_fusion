import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import statistics as st
import tracker_fusion.Fusion as fu
import tracker_fusion.Utils as ut


#def build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, image):
#    "Input: Image with 3D Boxes already drawn"
#    final_image = image.copy()
#    list_of_fused_objects = []
#    for match in matches:
#        fused_object = fu.FusedObject(list_of_2d_objects[match[1]].bbox, list_of_3d_objects[match[0]].bbox3d,
#                                   list_of_2d_objects[match[1]].category, list_of_3d_objects[match[0]].t,
#                                   list_of_2d_objects[match[1]].confidence)     
#                                                                              
#        cv2.putText(final_image, '{0:.2f} m'.format(fused_object.t[2]), (int(fused_object.bbox2d[0] + 15),int(fused_object.bbox2d[1] + 15)),
#        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 255), 1, cv2.LINE_AA)
#        cv2.putText(final_image, 'Confidence: {0:.2f}'.format(fused_object.confidence[0]),
#                     (int(fused_object.bbox2d[0] + 15), int(fused_object.bbox2d[1] + 30)),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 255), 1, cv2.LINE_AA)

        # cv2.putText(final_image, fused_object.class_, (int(fused_object.bbox2d[0]+15),int(fused_object.bbox2d[1]+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100,
        #255), 1, cv2.LINE_AA)
#    return final_image, list_of_fused_objects

def build_fused_object(list_of_2d_objects, list_of_3d_objects, matches, image):
    final_image = image.copy()
    list_of_fused_objects = []

    unmatched_lidar_boxes = [lidar_index for lidar_index in range(len(list_of_3d_objects)) if lidar_index not in [match[0] for match in matches]]
    unmatched_camera_boxes = [camera_index for camera_index in range(len(list_of_2d_objects)) if camera_index not in [match[1] for match in matches]]

    # Handle unmatched LiDAR boxes
    for lidar_index in unmatched_lidar_boxes:

        bbox = list_of_3d_objects[lidar_index].bbox2d
        category = list_of_3d_objects[lidar_index].class_
        confidence = 0.9
        list_of_fused_objects.append([bbox, category, confidence])

    # Handle matched boxes
    for match in matches:
        # Obtener las coordenadas de las cajas 2D y 3D
        bbox_2d = list_of_2d_objects[match[1]].bbox
        bbox_3d = list_of_3d_objects[match[0]].bbox2d
    
        # Calcular el promedio de las coordenadas
        avg_bbox = [
             (bbox_2d[0] + bbox_3d[0]) / 2,
             (bbox_2d[1] + bbox_3d[1]) / 2,
             (bbox_2d[2] + bbox_3d[2]) / 2,
             (bbox_2d[3] + bbox_3d[3]) / 2,
        ]
        avg_bbox = np.array(avg_bbox, dtype=np.float64)
        fused_object = fu.FusedObject(
            list_of_2d_objects[match[1]].bbox, list_of_3d_objects[match[0]].bbox3d,
            list_of_2d_objects[match[1]].category, list_of_3d_objects[match[0]].t,
            list_of_2d_objects[match[1]].confidence
        )
        cv2.putText(final_image, '{0:.2f} m'.format(fused_object.t[2]), (int(fused_object.bbox2d[0] + 15),int(fused_object.bbox2d[1] + 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(final_image, 'Confidence: {0:.2f}'.format(fused_object.confidence[0]),
                     (int(fused_object.bbox2d[0] + 15), int(fused_object.bbox2d[1] + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 255), 1, cv2.LINE_AA)
        list_of_fused_objects.append([avg_bbox, fused_object.class_, fused_object.confidence[0]])
      
        #print("fused_object.bbox2d: ",fused_object.bbox2d)
        #print("avg_bbox: ",avg_bbox)
    # Handle unmatched camera boxes
    for camera_index in unmatched_camera_boxes:
        
        bbox = list_of_2d_objects[camera_index].bbox
        category = list_of_2d_objects[camera_index].category
        with open("/home/chimuelo/Documents/fusion_deepsort/Code/Data/model/yolov4/coco.names", 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        class_ = classes[category]
        #clase = list_of_2d_objects[camera_index].clase
        confidence = list_of_2d_objects[camera_index].confidence
        confidence = confidence[0]
        list_of_fused_objects.append([bbox, class_, confidence])
    
    return final_image, list_of_fused_objects
    
def rectContains(rect, pt, shrink_factor=0.0):

    x_min = rect[0]
    y_min = rect[1]
    width = rect[2]
    height = rect[3]

    center_x = x_min + width * 0.5
    center_y = y_min + height * 0.5

    new_width = width * (1 - shrink_factor)
    new_height = height * (1 - shrink_factor)

    x1 = int(center_x - new_width * 0.5)
    y1 = int(center_y - new_height * 0.5)
    x2 = int(center_x + new_width * 0.5)
    y2 = int(center_y + new_height * 0.5)

    return x1 < pt[0] < x2 and y1 < pt[1] < y2


def filter_outliers(distances):
    inliers = []
    mu  = st.mean(distances)
    std = st.stdev(distances)
    for x in distances:
        if abs(x-mu) < std:
            # This is an INLIER
            inliers.append(x)
    return inliers


def get_best_distance(distances, technique="closest"):
    if technique == "closest":
        return min(distances)
    elif technique =="average":
        return st.mean(distances)
    elif technique == "random":
        return random.choice(distances)
    else:
        return st.median(sorted(distances))


def lidar_camera_fusion(pts_3D, pts_2D, detections, image):
    img_bis = image.copy()
    pred_bboxes = [detection[1] for detection in detections]
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    distances = []
    for box in pred_bboxes:
        distances = []
        for i in range(pts_2D.shape[0]):
            depth = pts_3D[i, 0]
            #if rectContains(box, pts_2D[i], 0.1):
            distances.append(depth)

                #color = cmap[int(510.0 / depth), :]
            cv2.circle(img_bis, (int(np.round(pts_2D[i, 0])), int(np.round(pts_2D[i, 1]))),
                       2, color=tuple((0,0,250)), thickness=-1, )
                #color = cmap[min(int(510.0 / depth), 255), :]

            

        h, w, _ = img_bis.shape
        if len(distances) > 2:
            distances = filter_outliers(distances)
            best_distance = get_best_distance(distances, technique="average")
            cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0] * w), int(box[1] * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0] * w), int(box[1] * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
        distances_to_keep = []

    return img_bis, distances

def associate(lidar_boxes, camera_boxes):
    """
    LiDAR boxes will represent the red bounding boxes
    Camera will represent the other bounding boxes
    Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
    """
    # Define a new IOU Matrix nxm with old and new boxes
    iou_matrix = np.zeros((len(lidar_boxes), len(camera_boxes)), dtype=np.float32)

    # Go through boxes and store the IOU value for each box
    # You can also use the more challenging cost but still use IOU as a reference for convenience (use as a filter only)
    for i, lidar_box in enumerate(lidar_boxes):
        for j, camera_box in enumerate(camera_boxes):          
            iou_matrix[i][j] = ut.get_iou(lidar_box, camera_box)     
    # Call for the Hungarian Algorithm
    hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
    hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

    # Create new unmatched lists for old and new boxes
    matches, unmatched_camera_boxes, unmatched_lidar_boxes = [], [], []

    # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched
    # Else: add the match
    for h in hungarian_matrix:
        if iou_matrix[h[0], h[1]] > 0.1:
            matches.append(h.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    #print("lidar : ",hungarian_matrix[:, 0]);print("camera : ",hungarian_matrix[:, 1])
     # Determine unmatched LiDAR boxes
    unmatched_lidar_boxes = [l for l in range(len(lidar_boxes)) if l not in matches[:, 0]]

    # Determine unmatched camera boxes
    unmatched_camera_boxes = [c for c in range(len(camera_boxes)) if c not in matches[:, 1]]
    
    return matches, np.array(unmatched_lidar_boxes), np.array(unmatched_camera_boxes)
