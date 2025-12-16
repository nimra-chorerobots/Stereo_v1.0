import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = r"D:\Chorerobotics\data_scene_flow\training"
LEFT_DIR  = os.path.join(DATA_ROOT, "image_2")
RIGHT_DIR = os.path.join(DATA_ROOT, "image_3")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_CFG     = os.path.join(BASE_DIR, "yolov3-tiny.cfg")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolov3-tiny.weights")
COCO_NAMES   = os.path.join(BASE_DIR, "coco.names")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# KITTI camera parameters (approx)
FOCAL_LENGTH = 721.5377
BASELINE = 0.54  # meters

# =========================================================
# LOAD YOLO-TINY
# =========================================================
def load_yolo():
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    with open(COCO_NAMES, "r") as f:
        classes = [c.strip() for c in f.readlines()]
    return net, out_layers, classes

# =========================================================
# PICK STEREO SAMPLE
# =========================================================
def pick_sample():
    imgs = sorted(glob.glob(os.path.join(LEFT_DIR, "*_10.png")))
    if not imgs:
        imgs = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
    return os.path.basename(imgs[0])

# =========================================================
# DEPTH COMPUTATION
# =========================================================
def compute_depth(left_gray, right_gray):
    stereo = cv2.StereoSGBM_create(
        numDisparities=256,
        blockSize=5,
        P1=8 * 3 * 5 * 5,
        P2=32 * 3 * 5 * 5,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0
    depth = np.zeros_like(disparity)
    mask = disparity > 0
    depth[mask] = (FOCAL_LENGTH * BASELINE) / disparity[mask]
    return disparity, depth

# =========================================================
# YOLO DETECTION
# =========================================================
def yolo_detect(net, out_layers, img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes, scores, class_ids = [], [], []
    for out in outputs:
        for det in out:
            if det[4] > 0.4:
                cls = np.argmax(det[5:])
                score = det[5:][cls]
                if score > 0.4:
                    cx, cy, bw, bh = det[0:4]
                    x = int((cx - bw/2) * w)
                    y = int((cy - bh/2) * h)
                    boxes.append([x, y, int(bw*w), int(bh*h)])
                    scores.append(float(score))
                    class_ids.append(cls)

    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.45)
    return [(boxes[i], class_ids[i], scores[i]) for i in idxs.flatten()] if len(idxs) else []

# =========================================================
# POINT CLOUD EXPORT
# =========================================================
def save_point_cloud(depth, img, path, stride=4):
    h, w = depth.shape
    with open(path, "w") as f:
        pts = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                z = depth[y, x]
                if 0 < z < 50:
                    X = (x - w/2) * z / FOCAL_LENGTH
                    Y = (y - h/2) * z / FOCAL_LENGTH
                    pts.append((X, Y, z, *img[y, x][::-1]))
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in pts:
            f.write("{} {} {} {} {} {}\n".format(*p))

# =========================================================
# MAIN
# =========================================================
def main():
    sample = pick_sample()
    left  = cv2.imread(os.path.join(LEFT_DIR, sample))
    right = cv2.imread(os.path.join(RIGHT_DIR, sample))

    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disparity, depth = compute_depth(grayL, grayR)

    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    depth_clip = np.clip(depth, 0, 50)
    depth_color = cv2.applyColorMap((depth_clip/50*255).astype(np.uint8), cv2.COLORMAP_TURBO)

    net, out_layers, classes = load_yolo()
    detections = yolo_detect(net, out_layers, left)

    semantic = left.copy()
    print("\nDetected Objects:")
    for box, cid, _ in detections:
        x,y,w,h = box
        roi = depth[y:y+h, x:x+w]
        roi = roi[(roi>0)&(roi<80)]
        d = np.median(roi) if roi.size else None
        label = classes[cid]
        if d:
            label += f" {d:.1f}m"
            print(label)
        cv2.rectangle(semantic,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(semantic,label,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    # 🔧 FIXED PLANNING OVERLAY
    planning = left.copy()
    mask = (depth > 0) & (depth < 10)
    planning[mask] = (
        planning[mask] * 0.3
        + np.array([0, 0, 255], dtype=np.uint8) * 0.7
    ).astype(np.uint8)

    free_space = np.zeros_like(depth, dtype=np.uint8)
    free_space[depth > 10] = 255

    save_point_cloud(depth, left, os.path.join(OUTPUT_DIR,"scene.ply"))

    fig, ax = plt.subplots(1,6, figsize=(30,6))
    titles = ["Stereo","Disparity","Metric Depth","Semantics","Planning","Free Space"]
    images = [left, disp_vis, depth_color, semantic, planning, free_space]

    for i in range(6):
        ax[i].imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB) if i!=5 else images[i], cmap="gray")
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
