import os, cv2, random, numpy as np, onnxruntime as ort, time

# ===== PATHS (pakai raw string) =====
MODEL_PATH = r"E:\ai-ml-portfolio-folders\Computer Vision\Box Recognition Method and Computing Device\best4.onnx"
IMG_PATH   = r"E:\ai-ml-portfolio-folders\Computer Vision\Box Recognition Method and Computing Device\04_0001.jpg"
ROI_PATH   = r"E:\ai-ml-portfolio-folders\Computer Vision\Box Recognition Method and Computing Device\roi_coordinates04New.txt"
OUT_PATH   = r"D:\PROJECT\PHA_project\new_yolov7\result2024\best500-04_2.jpg"

# ===== Validasi file =====
for p, label in [(MODEL_PATH,"MODEL"), (IMG_PATH,"IMAGE"), (ROI_PATH,"ROI")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{label} not found: {p}")

# ===== Providers aman (tanpa warning) =====
avail = ort.get_available_providers()
providers = [p for p in ("CUDAExecutionProvider","CPUExecutionProvider") if p in avail]

session = ort.InferenceSession(MODEL_PATH, providers=providers)

# ==== Ambil ukuran input model (N,C,H,W) ====
_, _, H, W = session.get_inputs()[0].shape

def letterbox(im, new_shape=(640,640), color=(114,114,114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    if not scaleup: r = min(r, 1.0)
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    if auto: dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top,bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left,right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top,bottom,left,right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

names = ['04','13','A','Y']
colors = {n: [random.randint(0,255) for _ in range(3)] for n in names}

# ==== Baca image dengan path yang benar ====
img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError(f"Failed to read image: {IMG_PATH}")

with open(ROI_PATH, 'r') as f:
    roi_x, roi_y, roi_w, roi_h = map(int, f.readline().strip().split())

cropped_img = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()

# ==== Preprocess sesuai ukuran model ====
image, ratio, dwdh = letterbox(cropped_img, new_shape=(H, W), auto=False)
im = image.transpose(2,0,1)[None].astype(np.float32) / 255.0

in_names  = [i.name for i in session.get_inputs()]
out_names = [i.name for i in session.get_outputs()]

start = time.time()
outputs = session.run(out_names, {in_names[0]: im})[0]
end = time.time()

# ==== Postprocess + gambar boks ====
min_w, min_h = 10, 10
images_with_detections = []

for (batch_id, x0, y0, x1, y1, cls_id, score) in outputs:
    if score < 0.1: 
        continue
    box = np.array([x0, y0, x1, y1])
    box -= np.array(dwdh*2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()
    bw, bh = box[2]-box[0], box[3]-box[1]
    if bw < min_w or bh < min_h:
        continue
    cls_id = int(cls_id)
    label = f"{names[cls_id]} {round(float(score),3)}"
    cv2.rectangle(cropped_img, box[:2], box[2:], colors[names[cls_id]], 2)
    cv2.putText(cropped_img, label, (box[0], max(0, box[1]-4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225,255,255), 2)
    images_with_detections.append(names[cls_id])

# ==== Hitung per kelas ====
y = 28
for cls in sorted(set(images_with_detections)):
    cnt = images_with_detections.count(cls)
    cv2.putText(cropped_img, f"Total of {cls} box = {cnt}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    y += 30

# ==== Simpan & tampilkan ROI yang sudah digambar ====
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
ok = cv2.imwrite(OUT_PATH, cropped_img)
print("Saved:", ok, "->", OUT_PATH)
print("[INFO] Detecting Time {:.6f} s".format(end - start))

cv2.imshow("Box (ROI)", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
