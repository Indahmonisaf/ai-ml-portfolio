import onnxruntime as ort
import numpy as np
import cv2
import time

# load onnx model
cuda = True
w = "D:/PHA_project/new_yolov7/box_binary/mobilenet.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

model = ort.InferenceSession(w, providers=providers)
# oepn video capture
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
video_capture = cv2.VideoCapture('D:/PHA_project/new_yolov7/bin_test(1).mp4')

predicted_labels = []
total_time = []

# Load the ROI coordinates
with open('D:/PHA_project/new_yolov7/roi_coordinatesbin2.txt', 'r') as file:
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