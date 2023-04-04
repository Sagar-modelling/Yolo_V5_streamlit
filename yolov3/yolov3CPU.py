import cv2
import numpy as np

def objectDetector(img):

    yolo = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg') #defining the yolo network
    classes = []

    with open('coco.names', 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = yolo.getLayerNames()
    #getUnconnectedOutLayers() function is used to obtain the indices of the output layers that will be used for object detection.
    output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()] #to get output layers indices

    colorRed = (0,0,255)
    colorGreen = (0,255,0)

    #Loading images
    #name = 'D:/OneDrive/OneDrive - Tata Insights and Quants/vision/yolo_objectDetection_imagesCPU/image.jpg'
    #img = cv2.imread(name)
    height, width, channels = img.shape

    # Detecting objects
    """
    Resizes the image to the required size specified in the function call, i.e., (416,416)
    Scales the pixel values to the range [0,1] 
    Subtracts the mean value of (0,0,0) from the image
    Transposes the image from (H,W,C) format to (C,H,W) format
    """
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)
    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences, 0.5,0.4)
    print(len(boxes))
    print(len(indexes))
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x,y), (x+w,y+h), colorGreen, 3)
            cv2.putText(img, label, (x,y+30), cv2.FONT_HERSHEY_PLAIN, 3, colorRed, 2)

    #resized_img = cv2.resize(img, (int(width/4), int(height/5)))
    #cv2.imshow('image', resized_img)
    #cv2.imwrite('output.jpg'+name,img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img
