# ============================================================
# STEREO VISION PIPELINE (DISPARITY → DEPTH → SEMANTICS)
# ============================================================
#
# WHAT THIS SCRIPT DOES (IN SIMPLE WORDS):
# ---------------------------------------
# 1. Reads LEFT and RIGHT stereo images (KITTI-style)
# 2. Finds HORIZONTAL pixel shift between them (called disparity)
# 3. Converts that shift into REAL distance (depth in meters)
# 4. Detects objects using YOLO
# 5. Combines geometry + semantics + planning visualization
#

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import glob
import cv2                    # OpenCV: stereo vision + image processing
import numpy as np             # Numerical computations
import matplotlib.pyplot as plt


# ============================================================
# DATA & MODEL PATHS
# ============================================================

DATA_ROOT = r"D:\Chorerobotics\data_scene_flow\training"

# Left and Right stereo images
LEFT_DIR  = os.path.join(DATA_ROOT, "image_2")
RIGHT_DIR = os.path.join(DATA_ROOT, "image_3")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO model files (for object detection)
YOLO_CFG     = os.path.join(BASE_DIR, "yolov3-tiny.cfg")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolov3-tiny.weights")
COCO_NAMES   = os.path.join(BASE_DIR, "coco.names")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CAMERA GEOMETRY (VERY IMPORTANT)
# ============================================================
# These values come from KITTI camera calibration

FOCAL_LENGTH = 721.5377   # focal length (pixels)
BASELINE = 0.54           # distance between left & right camera (meters)


# ============================================================
# LOAD YOLO OBJECT DETECTOR
# ============================================================

def load_yolo():
    """
    Loads YOLOv3-Tiny network for object detection
    """
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)

    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    with open(COCO_NAMES, "r") as f:
        classes = [c.strip() for c in f.readlines()]

    return net, out_layers, classes


# ============================================================
# PICK ONE STEREO IMAGE PAIR
# ============================================================

def pick_sample():
    """
    Picks one left image and assumes same name exists in right camera
    """
    imgs = sorted(glob.glob(os.path.join(LEFT_DIR, "*_10.png")))
    if not imgs:
        imgs = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))

    return os.path.basename(imgs[0])


# ============================================================
# DISPARITY & DEPTH COMPUTATION
# ============================================================

def compute_depth(left_gray, right_gray):
    """
    Computes:
    1) Disparity map  (horizontal pixel shift)
    2) Depth map      (distance in meters)

    Uses StereoSGBM (Semi-Global Block Matching)
    """

    stereo = cv2.StereoSGBM_create(

        # ---------------------------------------
        # Maximum horizontal shift (pixels)
        # Must be divisible by 16
        # Larger value → can see closer objects
        # ---------------------------------------
        numDisparities=256,

        # ---------------------------------------
        # Block size used for matching (pixels)
        # Small → more detail, more noise
        # Large → smoother, less detail
        # ---------------------------------------
        blockSize=5,

        # ---------------------------------------
        # Smoothness penalties
        # P1: small disparity changes
        # P2: large disparity jumps
        # These enforce surface continuity
    
        # ---------------------------------------
        ## p = 8 x channels x block size²
        P1=8 * 3 * 5 * 5,
        P2=32 * 3 * 5 * 5,
        
        # ---------------------------------------
        # Reject ambiguous matches
        # ---------------------------------------
        uniquenessRatio=10,

        # ---------------------------------------
        # Remove small noisy regions (speckles)
        # ---------------------------------------
        speckleWindowSize=100,
        speckleRange=2
    )

    # Compute disparity (raw result is scaled by 16)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Remove invalid values
    disparity[disparity <= 0] = 0

    # ========================================================
    # DEPTH FORMULA (VERY IMPORTANT)
    # ========================================================
    #
    #           depth = (focal_length × baseline)
    #                   -------------------------
    #                         disparity
    #
    # Where:
    # - focal_length = camera focal length
    # - baseline     = distance between cameras
    # - disparity    = horizontal pixel shift
    #
    # Closer object  → large disparity → small depth
    # Far object     → small disparity → large depth
    #
    # ========================================================

    depth = np.zeros_like(disparity)
    mask = disparity > 0
    depth[mask] = (FOCAL_LENGTH * BASELINE) / disparity[mask]

    return disparity, depth


# ============================================================
# YOLO OBJECT DETECTION
# ============================================================

def yolo_detect(net, out_layers, img):
    """
    Runs object detection on the LEFT image
    """
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (416,416), swapRB=True
    )

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


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # Pick one stereo pair
    sample = pick_sample()

    left  = cv2.imread(os.path.join(LEFT_DIR, sample))
    right = cv2.imread(os.path.join(RIGHT_DIR, sample))

    # Convert to grayscale for stereo matching
    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Compute disparity and depth
    disparity, depth = compute_depth(grayL, grayR)

    # Normalize disparity for visualization
    disp_vis = cv2.normalize(
        disparity, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Depth visualization (meters → color)
    depth_clip = np.clip(depth, 0, 50)
    depth_color = cv2.applyColorMap(
        (depth_clip/50*255).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )

    # Object detection
    net, out_layers, classes = load_yolo()
    detections = yolo_detect(net, out_layers, left)

    # Draw semantic boxes with depth
    semantic = left.copy()
    for box, cid, _ in detections:
        x,y,w,h = box
        roi = depth[y:y+h, x:x+w]
        roi = roi[(roi>0)&(roi<80)]
        d = np.median(roi) if roi.size else None
        label = classes[cid]
        if d:
            label += f" {d:.1f}m"
        cv2.rectangle(semantic,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(
            semantic, label, (x,y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1
        )

    # Planning overlay (close obstacles)
    planning = left.copy()
    mask = (depth > 0) & (depth < 10)
    planning[mask] = (
        planning[mask]*0.3 + np.array([0,0,255])*0.7
    ).astype(np.uint8)

    # Free space map
    free_space = np.zeros_like(depth, dtype=np.uint8)
    free_space[depth > 10] = 255

    # Display all stages
    fig, ax = plt.subplots(1,6, figsize=(30,6))
    titles = ["Stereo","Disparity","Metric Depth","Semantics","Planning","Free Space"]
    images = [left, disp_vis, depth_color, semantic, planning, free_space]

    for i in range(6):
        ax[i].imshow(
            cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
            if i!=5 else images[i], cmap="gray"
        )
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

