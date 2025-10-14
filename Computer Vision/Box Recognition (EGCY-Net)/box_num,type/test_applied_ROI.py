import cv2
import random
import numpy as np
import onnxruntime as ort

cuda = True
w = "D:/PHA_project/new_yolov7/best.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)

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


# Open video capture
video_capture = cv2.VideoCapture('D:/PHA_project/new_yolov7/test_data/Y_a.mp4')
#img = cv2.imread('0915_vid/000_rpi_v3_imx477_cam0.jpeg')

with open('roi_coordinates.txt', 'r') as file:
    roi_x, roi_y, roi_width, roi_height = map(int, file.readline().strip().split())

frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Define the codec and create VideoWriter object
# out = cv2.VideoWriter('A3_ROI_result.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width, frame_height))
# result = cv2.VideoWriter('filename.avi', 
#                         cv2.VideoWriter_fourcc(*'MJPG'),
#                         10, (frame_width, frame_height))

min_height_size = 10
min_width_size = 10
while True:
    ret, frame = video_capture.read()
    
    if not ret:
        break

    cropped_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    cropped_frame = cv2.rotate(cropped_frame, cv2.ROTATE_90_CLOCKWISE)
    # Perform inference on the frame
    image = cropped_frame.copy()
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
    ori_images = [cropped_frame.copy()]
    images_with_detections = []    
    
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        if score >= 0.1:  # Apply threshold
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            name = names[cls_id]
            color = colors[name]
            name += ' '+str(score)

            box_width = box[2] - box[0]
            box_height = box[3] - box[1]

            if box_width >= min_width_size and box_height >= min_height_size:

                cv2.rectangle(cropped_frame,box[:2],box[2:],color,2)
                cv2.putText(cropped_frame,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
                images_with_detections.append(name)
            print(box_width)
            print(box_height)

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
        cv2.putText(cropped_frame, total_label, (10, 30 + tambah), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tambah += 30

        #result.write(frame)
    
    # Display the processed frame
    cv2.imshow("Box", cropped_frame)
    keyCode = cv2.waitKey(10) & 0xFF
    # Stop the program on the ESC key or 'q'
    if keyCode == 27 or keyCode == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
# result.release()
cv2.destroyAllWindows()
