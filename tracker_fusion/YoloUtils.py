import cv2

def draw_yolo_detections(image, detections, color=(0,255,0), classes_to_draw=("person", "car")):
    img = image.copy()
    n = 0
    with open("/home/chimuelo/Documents/fusion_deepsort/Code/Data/model/yolov4/coco.names", 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    for detect in detections:
        bbox = detect[1]
        category = classes[int(detect[0])]
        #confidence = detect[2]
        #confidence = [ '%.2f' % elem for elem in confidence ]
        confidence = detect[2][0]  # Tomamos el primer elemento de la lista de confianza
        confidence = round(confidence, 2)

        #print(confidence)
        if category in classes_to_draw and confidence >= 0.52:
            n += 1 
            #print('clases: ', category)
            
            cv2.rectangle(img, bbox, color, 2)
            cv2.putText(img, f"({confidence})", (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    #print(n)    
    return img

