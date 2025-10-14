import cv2

cap = cv2.VideoCapture(r"E:\ai-ml-portfolio-folders\Computer Vision\Box Recognition Method and Computing Device\04_0001.jpg")
# Read the first frame
ret, frame = cap.read()

# Select the ROI using the first frame
roi = cv2.selectROI('Select ROI', frame, False, False)
cv2.destroyAllWindows()

# Save the coordinates
roi_x, roi_y, roi_width, roi_height = roi
with open(r'roi_coordinates04New.txt', 'w') as file:
    file.write(f"{roi_x} {roi_y} {roi_width} {roi_height}")
    


cap.release()
