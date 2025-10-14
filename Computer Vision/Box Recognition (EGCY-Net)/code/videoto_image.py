import cv2
import os

# Read the video from specified path
video_path = "D:/PHA_project/new_yolov7/test_data/video/Y3.mp4"
output_folder = "D:/PHA_project/new_yolov7/cropped_video/Y3_image"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
cam = cv2.VideoCapture(video_path)

# Get the frames per second (fps) of the video
fps = int(cam.get(cv2.CAP_PROP_FPS))

# Initialize variables
frame_count = 0
current_second = 0

while True:
    # Read a frame
    ret, frame = cam.read()

    if ret:
        if frame_count == fps * current_second:
            # Save the frame as an image
            image_name = os.path.join(output_folder, f"13_{current_second:04d}.jpg")
            cv2.imwrite(image_name, frame)
            print(f"Saved {image_name}")

            # Move to the next second
            current_second += 1

        frame_count += 1
    else:
        break

# Release video capture and close windows
cam.release()
cv2.destroyAllWindows()
