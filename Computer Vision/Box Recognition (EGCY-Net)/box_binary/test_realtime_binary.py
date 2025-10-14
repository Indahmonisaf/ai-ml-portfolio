import onnxruntime
import numpy as np
import cv2
import time

# load onnx model
model = onnxruntime.InferenceSession('models/mobilenet.onnx')

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    #display_width=960,
    #display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        #"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            #display_width,
            #display_height,
        )
    )

# oepn video capture
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

predicted_labels = []
total_time = []

# Load the ROI coordinates
with open('/home/pha/workspace/box_binary/ROI_binary/avg_roi.txt', 'r') as file:
    roi_x, roi_y, roi_width, roi_height = map(int, file.readline().strip().split())


while True:
    ret , frame = video_capture.read()

    if not ret:
        break
    cropped_roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    processed_frame =cv2.resize(cropped_roi, IMAGE_SIZE) / 255.0

    input_name = model.get_inputs()[0].name
    input_shape = model.get_inputs()[0].shape
    input_data = np.array(processed_frame).astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0) # demension

    output_name = model.get_outputs()[0].name
    start = time.time()
    output = model.run([output_name], {input_name: input_data})
    runtime = time.time() - start
    total_time.append(runtime)
    #print(runtime)

    label_id = np.argmax(output)
    label = 'No_Box' if label_id == 1 else 'Box'

    predicted_labels.append(label)
    label_text = 'Prediction: ' + str(label)
    
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

    cv2.putText(frame, label_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    #cv2.putText(frame, str(label), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)

    keyCode = cv2.waitKey(10) & 0xFF
    # Stop the program on the ESC key or 'q'
    if keyCode == 27 or keyCode == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# print(sum(total_time) / len(total_time))