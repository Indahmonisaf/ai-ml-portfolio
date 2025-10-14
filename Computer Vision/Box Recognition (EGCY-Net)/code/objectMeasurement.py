import cv2
import random
import numpy as np
import get_colour
import onnxruntime as ort

cuda = True
w = "D:/new_yolov7/best.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)

#Define object specific variables  
dist = 0
focal = 450
pixels = 30
width = 4

# Open video capture
video_capture = cv2.VideoCapture('Y.mp4')


#find the distance from then camera
def get_dist(rectange_params,image):
    #find no of pixels covered
    pixels = rectange_params[1][0]
    print(pixels)
    #calculate distance
    dist = (width*focal)/pixels
    
    #Wrtie n the image
    image = cv2.putText(image, 'Distance from Camera in CM :', org, font,  
       1, color, 2, cv2.LINE_AA)

    image = cv2.putText(image, str(dist), (110,50), font,  
       fontScale, color, 1, cv2.LINE_AA)

    return image

#basic constants for opencv Functs
kernel = np.ones((3,3),'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (0,20)  
fontScale = 0.6 
color = (0, 0, 255) 
thickness = 2

cv2.namedWindow('Object Dist Measure ',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Dist Measure ', 700,600)


# Rest of your functions and code...
def count(founded_classes, im0):
    model_values = []
    aligns = im0.shape
    align_bottom = aligns[0]
    align_right = (aligns[1] / 1.7)

    for k, v in founded_classes.items():
        a = f"{k} = {v}"
        model_values.append(v)
        align_bottom = align_bottom - 35
        cv2.putText(im0, str(a), (int(align_right), align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (45, 255, 255), 1, cv2.LINE_AA)

def letterbox(im, new_shape=(256, 256), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

# List of class names and colors
names = ['04', '13', 'A', 'Y']
colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names}



frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('Y_result.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width, frame_height))



while True:
    ret, frame = video_capture.read()
    
    

    if not ret:
        break

    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #predefined mask for green colour detection
    lower = np.array([37, 51, 24])
    upper = np.array([83, 104, 131])
    mask = cv2.inRange(hsv_img, lower, upper)
     


    #Remove Extra garbage from image
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations = 5)


    #find the histogram
    
    cont,hei = cv2.findContours(d_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:1]
    
    # Perform inference on the frame
    image = frame.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255

    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: im}

    # ONNX inference
    outputs = session.run(outname, inp)[0]
    

    ori_images = [frame.copy()]
    class_ids = []
    images_with_detections = []
    confidences = []
    boxes = []

    for cnt in cont:
        #check for contour area
        if (cv2.contourArea(cnt)>100 and cv2.contourArea(cnt)<306000):
            
            for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
                if score >= 0.1:  # Apply threshold
                    image = ori_images[int(batch_id)]
                    box = np.array([x0, y0, x1, y1])
                    #box -= np.array(dwdh) * 2
                    box /= ratio
                    box = box.round().astype(np.int32).tolist()
                    cls_id = int(cls_id)
                    score = round(float(score), 3)
                    name = names[cls_id]
                    color = colors[name]
                    name += ' ' + str(score)

                    # Draw bounding box and label
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(image, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)

                    #Draw a rectange on the contour
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect) 
                    box = np.int0(box)
                    cv2.drawContours(image,[box], -1,(255,0,0),3)
            
                    img = get_dist(rect,image)
                    images_with_detections.append(name)

    # Count the occurrences of each class
    class_counts = {}
    for name in images_with_detections:
        class_name, _ = name.split(' ')
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    tambah = 0
    for class_name, count in class_counts.items():
        total_label = f"Total of {class_name} box = {count}"
        cv2.putText(frame, total_label, (10, 30 + tambah), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        tambah += 30

    out.write(frame)
    
    # Display the processed frame
    cv2.imshow('Object Dist Measure ',image)
    cv2.imshow("Box", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

# Release video capture and close windows
video_capture.release()
out.release()
cv2.destroyAllWindows()
