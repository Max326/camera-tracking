import cv2
import os
import sys
import numpy as np
import pandas as pd
import glob

# 1. Get the directory where THIS script resides (src/tracking)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels to get the Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# 3. Define Data Paths relative to Project Root
# Based on ReadMe, UAV20L is better for 1:1 mapping (full sequences) [cite: 7, 8]
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "UAV123")
ANNO_PATH = os.path.join(DATA_ROOT, "anno", "UAV20L")  # Use UAV20L for full sequences
SEQ_PATH  = os.path.join(DATA_ROOT, "data_seq", "UAV123")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

print(f"Project Root: {PROJECT_ROOT}")
print(f"Looking for annotations in: {ANNO_PATH}")

# Available OpenCV Trackers
TRACKERS = {
    "CSRT": cv2.TrackerCSRT_create,
    "KCF": cv2.legacy.TrackerKCF_create,   # Requires opencv-contrib-python
    "MOSSE": cv2.legacy.TrackerMOSSE_create # Requires opencv-contrib-python
}

def load_ground_truth(anno_file):
    """
    Reads UAV123 txt files. 
    Format is usually: x,y,w,h (comma or space separated).
    NaN indicates out-of-view/occlusion.
    """
    with open(anno_file, 'r') as f:
        lines = f.readlines()
    
    gt_boxes = []
    for line in lines:
        # Replace commas with spaces to handle both formats
        parts = line.replace(',', ' ').strip().split()
        try:
            # Parse numbers
            box = [float(p) for p in parts]
            # Check for NaN (occlusion defined in ReadMe)
            if np.isnan(box).any():
                gt_boxes.append(None)
            else:
                gt_boxes.append(box) # x, y, w, h
        except ValueError:
            gt_boxes.append(None)
            
    return gt_boxes

def calculate_iou(boxA, boxB):
    """Computes Intersection over Union."""
    if boxA is None or boxB is None: return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def run_benchmark(seq_name, tracker_name):
    print(f"Processing {seq_name} with {tracker_name}...")
    
    # 1. Load Annotations
    anno_file = os.path.join(ANNO_PATH, f"{seq_name}.txt")
    gt_boxes = load_ground_truth(anno_file)
    
    # 2. Load Images
    # UAV123 folder structure: data_seq/UAV123/seq_name/img00001.jpg
    img_folder = os.path.join(SEQ_PATH, seq_name)
    images = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
    
    if not images:
        print(f"Error: No images found for {seq_name}")
        return

    # 3. Initialize Tracker
    tracker = TRACKERS[tracker_name]()
    
    # Read first valid frame
    frame = cv2.imread(images[0])
    first_box = gt_boxes[0]
    
    # UAV123 allows starting with NaN? usually no, but safety check:
    if first_box is None:
        print("Skipping: Sequence starts with occlusion.")
        return

    # Init tracker with Ground Truth (x, y, w, h)
    tracker.init(frame, tuple(first_box))
    
    results = []
    
    # 4. Loop through frames
    for i, img_path in enumerate(images):
        if i == 0: continue # Skip init frame
        
        frame = cv2.imread(img_path)
        
        # Start Timer
        timer = cv2.getTickCount()
        success, box = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Ground Truth for this frame
        gt = gt_boxes[i] if i < len(gt_boxes) else None
        
        # Calculate Metrics
        iou = 0.0
        center_error = 0.0
        
        if success:
            if gt is not None:
                iou = calculate_iou(box, gt)
                # Center Error
                c_track = (box[0] + box[2]/2, box[1] + box[3]/2)
                c_gt = (gt[0] + gt[2]/2, gt[1] + gt[3]/2)
                center_error = np.linalg.norm(np.array(c_track) - np.array(c_gt))
            else:
                # GT is NaN (Occluded), but tracker thinks it sees something.
                # In standard benchmarks, we usually ignore this frame for accuracy
                # but might penalize for not reporting "lost".
                iou = -1 # Mark as ignored
        
        results.append({
            "Frame": i,
            "FPS": fps,
            "IoU": iou,
            "CenterError": center_error,
            "GT_Occluded": gt is None
        })
        
    # Save Results
    df = pd.DataFrame(results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/{seq_name}_{tracker_name}.csv", index=False)
    
    print(f"Done. Avg FPS: {df['FPS'].mean():.2f}, Avg IoU: {df[df['IoU'] >= 0]['IoU'].mean():.2f}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Example: Run on one sequence
    # You can loop through all .txt files in ANNO_PATH to run batch
    run_benchmark("car1", "KCF")
    run_benchmark("car1", "CSRT")