import cv2
import random
import numpy as np
import onnxruntime as ort

# Use CUDA if available
cuda = True
w = "D:/PROJECT/PHA_project/new_yolov7/RIS_code/best.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)

# Letterbox function to resize and pad image
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

# List of class names and colors
names = ['other', 'fire', 'smoke']
colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names}

# Open video capture
video_capture = cv2.VideoCapture('D:/PROJECT/PHA_project/new_yolov7/RIS_code/IMG_1080.MP4')

frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Define the codec and create VideoWriter object
result = cv2.VideoWriter('yolo-csp.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (frame_width, frame_height))

min_height_size = 10
min_width_size = 10

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

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

    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        if score >= 0.1:  # Apply threshold
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            name = names[cls_id]
            color = colors[name]
            name += ' ' + str(round(float(score), 3))

            box_width = box[2] - box[0]
            box_height = box[3] - box[1]

            if box_width >= min_width_size and box_height >= min_height_size:
                cv2.rectangle(frame, box[:2], box[2:], color, 2)
                cv2.putText(frame, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)

    result.write(frame)

    # Display the processed frame
    cv2.imshow("Box", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

# Release video capture and close windows
video_capture.release()
result.release()
cv2.destroyAllWindows()
