import cv2
import os
import sys
import numpy as np
import pandas as pd
import glob

# --- CONFIGURATION ---
# Set to TRUE to watch the video (slows down processing)
# Set to FALSE when you want to benchmark the whole dataset fast
VISUALIZE = True 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "UAV123")
ANNO_PATH = os.path.join(DATA_ROOT, "anno", "UAV20L")
SEQ_PATH  = os.path.join(DATA_ROOT, "data_seq", "UAV123")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

TRACKERS = {
    "MIL": cv2.legacy.TrackerMIL_create,
    "CSRT": cv2.TrackerCSRT_create,
    "KCF": cv2.legacy.TrackerKCF_create,
    "MOSSE": cv2.legacy.TrackerMOSSE_create,
    "TLD": cv2.legacy.TrackerTLD_create,
    # "BOOSTING": cv2.legacy.TrackerBOOSTING_create
    "BOOSTING": cv2.legacy.TrackerBoosting_create,
    "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
    # The following trackers require extra model files
    # "GOTURN": cv2.TrackerGOTURN_create,
    # "DaSiamRPN": cv2.TrackerDaSiamRPN_create
}

def load_ground_truth(anno_file):
    with open(anno_file, 'r') as f:
        lines = f.readlines()
    gt_boxes = []
    for line in lines:
        parts = line.replace(',', ' ').strip().split()
        try:
            box = [float(p) for p in parts]
            if np.isnan(box).any():
                gt_boxes.append(None)
            else:
                gt_boxes.append(box)
        except ValueError:
            gt_boxes.append(None)
    return gt_boxes

def calculate_iou(boxA, boxB):
    if boxA is None or boxB is None: return 0.0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def run_benchmark(seq_name, tracker_name):
    print(f"--- Processing {seq_name} : {tracker_name} ---")
    
    # 1. Load Data
    anno_file = os.path.join(ANNO_PATH, f"{seq_name}.txt")
    if not os.path.exists(anno_file):
        print(f"Missing annotation: {anno_file}")
        return

    gt_boxes = load_ground_truth(anno_file)
    img_folder = os.path.join(SEQ_PATH, seq_name)
    if not os.path.exists(img_folder):
        img_folder = os.path.join(SEQ_PATH, seq_name.split('_')[0]) # Try parent folder
    
    images = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
    if not images:
        print("No images found.")
        return

    # 2. Initialize Tracker
    tracker = TRACKERS[tracker_name]()
    first_frame = cv2.imread(images[0])
    first_box_float = gt_boxes[0]

    if first_box_float is None:
        print("Starts with occlusion. Skipping.")
        return

    # FIX: Convert float box to int tuple for OpenCV init
    first_box_int = tuple(map(int, first_box_float))
    tracker.init(first_frame, first_box_int)
    
    results = []
    max_frames = min(len(images), len(gt_boxes))

    # 3. Main Loop
    for i in range(1, max_frames):
        frame = cv2.imread(images[i])
        if frame is None: break

        timer = cv2.getTickCount()
        success, box = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        gt = gt_boxes[i]
        iou = 0.0
        
        # Calculate IoU if GT exists
        if gt is not None:
            iou = calculate_iou(box, gt)

        # --- VISUALIZATION BLOCK ---
        if VISUALIZE:
            # Draw Tracker (Red)
            if success:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
                cv2.putText(frame, tracker_name, (p1[0], p1[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Draw Ground Truth (Green)
            if gt is not None:
                p1_g = (int(gt[0]), int(gt[1]))
                p2_g = (int(gt[0] + gt[2]), int(gt[1] + gt[3]))
                cv2.rectangle(frame, p1_g, p2_g, (0, 255, 0), 2, 1)
                cv2.putText(frame, "GT", (p1_g[0], p1_g[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Dashboard
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"IoU: {iou:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Tracking Benchmark", frame)
            
            # Press ESC to stop this sequence early
            if cv2.waitKey(1) & 0xFF == 27:
                print("User skipped sequence.")
                break
        # ---------------------------

        results.append({"Frame": i, "FPS": fps, "IoU": iou})

    cv2.destroyAllWindows()
    
    # Stats
    df = pd.DataFrame(results)
    avg_iou = df['IoU'].mean() if not df.empty else 0
    avg_fps = df['FPS'].mean() if not df.empty else 0
    print(f"Result: Avg IoU: {avg_iou:.2f} | Avg FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    # Test just ONE difficult sequence first to see the difference
    run_benchmark("bird1", "BOOSTING")
    run_benchmark("bird1", "MEDIANFLOW")
    # run_benchmark("bird1", "MIL") # Too slow
    run_benchmark("bird1", "CSRT")
    run_benchmark("bird1", "KCF")
    run_benchmark("bird1", "MOSSE")
    # run_benchmark("bird1", "TLD") # Very unstable, very slow