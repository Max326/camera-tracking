import cv2
import os
import sys
import numpy as np
import pandas as pd
import glob
import math
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive to avoid thread/interactivity crashes
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- CONFIGURATION ---
VISUALIZE = True 
# Save a big CSV with every frame from every tracker?
SAVE_DETAILED_LOGS = False 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/datset", "UAV123_640x480")
ANNO_PATH_UAV123 = os.path.join(DATA_ROOT, "anno", "UAV123")
ANNO_PATH_UAV20L = os.path.join(DATA_ROOT, "anno", "UAV20L")
SEQ_PATH  = os.path.join(DATA_ROOT, "data_seq", "UAV123")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results-rpi-scaled")

# --- YOLO WRAPPER ---
class YOLOTrackerWrapper:
    def __init__(self, model_path='yolov8n.pt', tracker_config='botsort.yaml'):
        # Explicitly set task='detect' to suppress warnings for NCNN/TFLite models
        self.model = YOLO(model_path, task='detect')
        self.tracker_config = tracker_config
        self.target_id = None
        self.last_box = None
        self.lost_frames = 0
        self.max_lost_frames = 60 # Allow 2 seconds (at 30fps) to recover

    def init(self, frame, box):
        # box is (x, y, w, h)
        # Run tracking on the first frame to get IDs
        results = self.model.track(frame, persist=True, tracker=self.tracker_config, verbose=False)
        
        # FIX: Check if boxes.id is None explicitly
        if not results or results[0].boxes.id is None:
            return False

        # Find the detection closest to the init box to lock onto the ID
        min_dist = float('inf')
        best_id = None
        
        # Convert init box to center
        gt_center = (box[0] + box[2]/2, box[1] + box[3]/2)

        boxes = results[0].boxes.xywh.cpu().numpy() # center_x, center_y, w, h
        ids = results[0].boxes.id.cpu().numpy()

        for i, det_box in enumerate(boxes):
            # det_box is [cx, cy, w, h]
            dist = math.sqrt((det_box[0] - gt_center[0])**2 + (det_box[1] - gt_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_id = ids[i]
                # Store box in x,y,w,h format for consistency
                self.last_box = (det_box[0] - det_box[2]/2, det_box[1] - det_box[3]/2, det_box[2], det_box[3])

        self.target_id = best_id
        self.last_box = (det_box[0] - det_box[2]/2, det_box[1] - det_box[3]/2, det_box[2], det_box[3])
        self.lost_frames = 0
        return True

    def update(self, frame):
        if self.target_id is None:
            return False, (0,0,0,0)

        results = self.model.track(frame, persist=True, tracker=self.tracker_config, verbose=False)
        
        if not results or results[0].boxes.id is None:
            self.lost_frames += 1
            return False, self.last_box # Return last known position

        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        # 1. Try to find the existing target_id
        idx = np.where(ids == self.target_id)[0]
        
        if len(idx) > 0:
            # Found it!
            i = idx[0]
            det_box = boxes[i]
            x = det_box[0] - det_box[2]/2
            y = det_box[1] - det_box[3]/2
            w = det_box[2]
            h = det_box[3]
            self.last_box = (x, y, w, h)
            self.lost_frames = 0
            return True, self.last_box
        
        # 2. If ID not found, try to recover if we haven't been lost for too long
        if self.lost_frames < self.max_lost_frames:
            # Look for a new ID that overlaps significantly with our last known box
            best_iou = 0
            best_new_id = None
            best_new_box = None
            
            last_box_xywh = self.last_box # (x, y, w, h)
            
            for i, det_box in enumerate(boxes):
                # det_box is (cx, cy, w, h) -> convert to (x, y, w, h)
                current_box = (det_box[0] - det_box[2]/2, det_box[1] - det_box[3]/2, det_box[2], det_box[3])
                
                iou = calculate_iou(last_box_xywh, current_box)
                if iou > 0.2: # Strict threshold to avoid jumping to wrong object
                    if iou > best_iou:
                        best_iou = iou
                        best_new_id = ids[i]
                        best_new_box = current_box
            
            if best_new_id is not None:
                print(f"ID Switch: {self.target_id} -> {best_new_id} (IoU: {best_iou:.2f})")
                self.target_id = best_new_id
                self.last_box = best_new_box
                self.lost_frames = 0
                return True, self.last_box

        self.lost_frames += 1
        return False, self.last_box # Target lost in this frame
    
class HybridTrackerWrapper:
    def __init__(self, model_path='yolo11n.pt', detection_interval=10, tracker="KCF", conf_threshold=0.6):
        self.model = YOLO(model_path, task='detect')
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False) # model warm-up

        self.tracker_name = tracker
        self.tracker = TRACKERS[tracker]()
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.last_box = None
        self.is_tracking = False
        self.target_class = None
        
        # Appearance Confidence
        self.conf_threshold = conf_threshold
        self.ref_hist = None

    def get_hist(self, frame, box):
        """Extracts HSV histogram from the bounding box area."""
        x, y, w, h = map(int, box)
        h_img, w_img = frame.shape[:2]
        
        # Clip to image bounds
        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)
        
        if w <= 0 or h <= 0: return None
        
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate Hue histogram (0-179)
        hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def init(self, frame, box):
        self.last_box = box
        self.tracker.init(frame, box)
        self.is_tracking = True
        self.frame_count = 0
        
        # Initialize appearance reference
        self.ref_hist = self.get_hist(frame, box)

        results = self.model(frame, verbose=False)
        if results and results[0].boxes:
            boxes = results[0].boxes.xywh.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            best_iou = 0
            for i, det_box in enumerate(boxes):
                # Convert center-xywh to top-left-xywh
                current_box = (det_box[0] - det_box[2]/2, det_box[1] - det_box[3]/2, det_box[2], det_box[3])
                iou = calculate_iou(box, current_box)
                
                # If the initial box overlaps significantly with a detection, lock onto that class
                if iou > 0.5 and iou > best_iou:
                    best_iou = iou
                    self.target_class = classes[i]
        
        if self.target_class is not None:
            print(f"Initialized tracking for class ID: {int(self.target_class)}")
            
        return True

    def update(self, frame):
        self.frame_count += 1
        
        # 1. Always update the tracker (Fast)
        success, box = self.tracker.update(frame)
        
        # 2. Calculate Confidence
        confidence = 0.0
        if success and self.ref_hist is not None:
            cur_hist = self.get_hist(frame, box)
            if cur_hist is not None:
                # Compare Histogram: Correlation (1.0 = perfect match)
                confidence = cv2.compareHist(self.ref_hist, cur_hist, cv2.HISTCMP_CORREL)
        
        if success:
            self.last_box = box
        
        # 3. Run YOLO if:
        #    a) Tracking entirely failed
        #    b) Confidence (Appearance) dropped below threshold
        #    c) (Optional) Helper periodic check if you still want it, otherwise we rely on confidence
        
        need_correction = not success or (confidence < self.conf_threshold)

        if need_correction:
            # Restrict detection to the target class if known
            target_classes = [int(self.target_class)] if self.target_class is not None else None
            results = self.model(frame, verbose=False, classes=target_classes)
            
            best_match_box = None
            best_metric = 0 if success else float('inf') # IoU (max) or Distance (min)
            
            if results and results[0].boxes:
                boxes = results[0].boxes.xywh.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for i, det_box in enumerate(boxes):
                    # CLASS CHECK: Ignore detections that don't match our target class
                    if self.target_class is not None and classes[i] != self.target_class:
                        continue

                    # Convert center-xywh to top-left-xywh
                    current_box = (det_box[0] - det_box[2]/2, det_box[1] - det_box[3]/2, det_box[2], det_box[3])
                    
                    if success:
                        # CORRECTION: Use IoU
                        iou = calculate_iou(box, current_box)
                        if iou > 0.0 and iou > best_metric:
                            best_metric = iou
                            best_match_box = current_box
                    else:
                        # RECOVERY: Use Distance to last known position
                        if self.last_box:
                            lx, ly, lw, lh = self.last_box
                            cx, cy, cw, ch = current_box
                            l_center = (lx + lw/2, ly + lh/2)
                            c_center = (cx + cw/2, cy + ch/2)
                            dist = math.sqrt((l_center[0] - c_center[0])**2 + (l_center[1] - c_center[1])**2)
                            
                            if dist < best_metric:
                                best_metric = dist
                                best_match_box = current_box

            # If detector found the object, re-initialize the tracker
            if best_match_box is not None:
                # print(f"Correction/Recovery applied at frame {self.frame_count}")
                self.tracker = TRACKERS[self.tracker_name]()
                self.tracker.init(frame, tuple(map(int, best_match_box)))
                return True, best_match_box

        return success, box

# --- TRACKER FACTORY ---
def create_tracker(name):
    match name:
        case "YOLOv8-BoT":
            return YOLOTrackerWrapper(tracker_config="botsort.yaml")
        case "YOLOv8-Byte":
            return YOLOTrackerWrapper(tracker_config="bytetrack.yaml")
        case "YOLOv11-BoT":
            return YOLOTrackerWrapper(model_path="yolo11n.pt", tracker_config="botsort.yaml")
        case "YOLOv11-Byte":
            return YOLOTrackerWrapper(model_path="yolo11n.pt", tracker_config="bytetrack.yaml")
        case "YOLOv8-NCNN-BoT":
            return YOLOTrackerWrapper(model_path="yolov8n_ncnn_model", tracker_config="botsort.yaml")
        case "YOLOv8-NCNN-Byte":
            return YOLOTrackerWrapper(model_path="yolov8n_ncnn_model", tracker_config="bytetrack.yaml")
        case "YOLOv11-NCNN-BoT":
            return YOLOTrackerWrapper(model_path="yolo11n_ncnn_model", tracker_config="botsort.yaml")
        case "YOLOv11-NCNN-Byte":
            return YOLOTrackerWrapper(model_path="yolo11n_ncnn_model", tracker_config="bytetrack.yaml")
        case "RTDETR-BoT":
            return YOLOTrackerWrapper(model_path="rtdetr-l.pt", tracker_config="botsort.yaml")
        case "YOLOv8-NCNN+KCF-adaptive":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=10, tracker="KCF")
        case "YOLOv8-NCNN+MOSSE-adaptive":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=10, tracker="MOSSE")
        case "YOLOv8-NCNN+CSRT-90":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=90, tracker="CSRT")
        case "YOLOv11-NCNN+KCF-30":
            return HybridTrackerWrapper(model_path='yolo11n_ncnn_model', detection_interval=30, tracker="KCF")
        case "YOLOv11-NCNN+MOSSE-30":
            return HybridTrackerWrapper(model_path='yolo11n_ncnn_model', detection_interval=30, tracker="MOSSE")
        case "YOLOv11-NCNN+CSRT-90":
            return HybridTrackerWrapper(model_path='yolo11n_ncnn_model', detection_interval=90, tracker="CSRT")
        case "YOLOv8-NCNN+BOOSTING-30":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=30, tracker="BOOSTING")
        case "YOLOv8-NCNN+MEDIANFLOW-15":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=15, tracker="MEDIANFLOW")
        case "YOLOv8-NCNN+BOOSTING-60":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=60, tracker="BOOSTING")
        case "YOLOv8-NCNN+MEDIANFLOW-60":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=60, tracker="MEDIANFLOW")
        case "YOLOv8-NCNN+KCF-15":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=15, tracker="KCF")
        case "YOLOv8-NCNN+MOSSE-15":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=15, tracker="MOSSE")
        case "YOLOv8-NCNN+CSRT-15":
            return HybridTrackerWrapper(model_path='yolov8n_ncnn_model', detection_interval=60, tracker="CSRT")
        case "YOLOv11-NCNN+KCF-15":
            return HybridTrackerWrapper(model_path='yolo11n_ncnn_model', detection_interval=15, tracker="KCF")
        case "YOLOv11-NCNN+MOSSE-15":
            return HybridTrackerWrapper(model_path='yolo11n_ncnn_model', detection_interval=15, tracker="MOSSE")
        case "YOLOv11-NCNN+CSRT-60":
            return HybridTrackerWrapper(model_path='yolo11n_ncnn_model', detection_interval=60, tracker="CSRT")
        case _:
            return TRACKERS[name]()

TRACKERS = {
    "MIL": cv2.legacy.TrackerMIL_create,
    "CSRT": cv2.legacy.TrackerCSRT_create,
    "KCF": cv2.legacy.TrackerKCF_create,
    "MOSSE": cv2.legacy.TrackerMOSSE_create,
    "TLD": cv2.legacy.TrackerTLD_create,
    "BOOSTING": cv2.legacy.TrackerBoosting_create,
    "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
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
    denominator = boxAArea + boxBArea - interArea
    if denominator == 0: return 0.0
    return interArea / float(denominator)

def get_center(box):
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

def calculate_center_error(box_track, box_gt):
    if box_track is None or box_gt is None: return -1.0
    c_t = get_center(box_track)
    c_g = get_center(box_gt)
    # Euclidean distance
    dist = math.sqrt((c_t[0] - c_g[0])**2 + (c_t[1] - c_g[1])**2)
    return dist

def run_benchmark(seq_name, tracker_name):
    print(f"--- Processing {seq_name} : {tracker_name} ---")
    
    # 1. Load Data
    # Try UAV123 first
    anno_file = os.path.join(ANNO_PATH_UAV123, f"{seq_name}.txt")
    if not os.path.exists(anno_file):
        # Try UAV20L
        anno_file = os.path.join(ANNO_PATH_UAV20L, f"{seq_name}.txt")
        if not os.path.exists(anno_file):
            print(f"Missing annotation: {anno_file} (checked UAV123 and UAV20L)")
            return None

    gt_boxes = load_ground_truth(anno_file)
    img_folder = os.path.join(SEQ_PATH, seq_name)
    if not os.path.exists(img_folder):
        img_folder = os.path.join(SEQ_PATH, seq_name.split('_')[0])
    
    images = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
    if not images:
        print("No images found.")
        return None

    # 2. Initialize Tracker
    # tracker = TRACKERS[tracker_name]() # OLD
    tracker = create_tracker(tracker_name)
    
    first_frame = cv2.imread(images[0])
    first_box_float = gt_boxes[0]

    if first_box_float is None:
        print("Starts with occlusion. Skipping.")
        return None

    first_box_int = tuple(map(int, first_box_float))
    tracker.init(first_frame, first_box_int)
    
    frame_data = []
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
        center_error = -1.0 # -1 means invalid/occluded
        
        if gt is not None:
            iou = calculate_iou(box, gt)
            center_error = calculate_center_error(box, gt)

        # --- VISUALIZATION BLOCK ---
        if VISUALIZE:
            if success:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
                cv2.putText(frame, tracker_name, (p1[0], p1[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if gt is not None:
                p1_g = (int(gt[0]), int(gt[1]))
                p2_g = (int(gt[0] + gt[2]), int(gt[1] + gt[3]))
                cv2.rectangle(frame, p1_g, p2_g, (0, 255, 0), 2, 1)

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"IoU: {iou:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if center_error != -1:
                cv2.putText(frame, f"Err: {center_error:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Tracking Benchmark", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("User skipped sequence.")
                break
        # ---------------------------

        frame_data.append({
            "Sequence": seq_name,
            "Tracker": tracker_name,
            "Frame": i,
            "FPS": fps,
            "IoU": iou,
            "CenterError": center_error
        })

    cv2.destroyAllWindows()
    return pd.DataFrame(frame_data)

def calculate_summary(df):
    """Calculates SOT Metrics: Precision (CLE < 20px), Success (IoU > 0.5), and AUC"""
    valid_df = df[df['CenterError'] != -1]
    successful_frames = df[df['IoU'] > 0]
    valid_fps = successful_frames['FPS'].mean() if not successful_frames.empty else 0.0
    
    # Calculate success rate at 100 thresholds from 0 to 1
    thresholds = np.linspace(0, 1, 101)
    success_rates = []
    for thresh in thresholds:
        success_rate = (df['IoU'] > thresh).mean()
        success_rates.append(success_rate)
    auc = np.mean(success_rates) # Approximate Area Under Curve
    # --------------------------

    summary = {
        "Tracker": df['Tracker'].iloc[0],
        "Sequence": df['Sequence'].iloc[0],
        "Avg FPS": df['FPS'].mean(),
        "Valid FPS": valid_fps,
        "Avg IoU": df['IoU'].mean(),
        "Precision (CLE < 20px)": (valid_df['CenterError'] < 20).mean(),
        "Success Rate (IoU > 0.5)": (df['IoU'] > 0.5).mean(),
        "AUC": auc
    }
    return summary

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define which trackers and sequences to run
    trackers_to_test = ["KCF"]
    #                     "BOOSTING", "MEDIANFLOW", "MIL", "TLD", "CSRT", "KCF", "MOSSE", "YOLOv11-Byte", "YOLOv8-Byte", 
    #                     "YOLOv11-BoT", "YOLOv8-BoT"]
    # trackers_to_test = ["CSRT", "KCF", "MOSSE", "MIL", "MEDIANFLOW", "BOOSTING", "TLD"]
    # trackers_to_test = ["YOLOv8-NCNN-BoT", "YOLOv8-NCNN-Byte", "YOLOv11-NCNN-Byte", "YOLOv11-NCNN-BoT"] 
    # trackers_to_test = ["BOOSTING", "MEDIANFLOW", "MIL", "TLD"]
    
    sequences_to_test = ["bike1", "bike3", "boat1", "boat2", "boat3", "car1", "car2", "car3", "car4",
                        "car5", "car6", "car7", "car8", "car16", "car17", "car18", "person2", "person3", 
                         "truck1", "truck2", "truck3", "wakeboard1", "wakeboard2", "wakeboard3"]
    # sequences_to_test = ["car5", "car6", "car7", "car8", "car16", "car17", "car18", "person2", "person3", 
    #                      "truck1", "truck2", "truck3", "wakeboard1", "wakeboard2", "wakeboard3"]
    
    all_frame_results = []
    summary_results = []

    for seq in sequences_to_test:
        seq_summary_results = [] # Reset for each sequence
        seq_frame_results = []   # Reset for each sequence

        for tracker_name in trackers_to_test:
            df = run_benchmark(seq, tracker_name)
            
            if df is not None and not df.empty:
                # 1. Collect Detailed Data
                seq_frame_results.append(df)
                all_frame_results.append(df) # Keep for global detailed log if needed
                
                # 2. Calculate Summary for this run
                summ = calculate_summary(df)
                seq_summary_results.append(summ)
                summary_results.append(summ) # Keep for global summary if needed
                print(f"Finished {tracker_name} on {seq} -> Precision: {summ['Precision (CLE < 20px)']:.2%}")

        # --- SAVE PER-SEQUENCE OUTPUTS ---
        if seq_summary_results:
            # 1. The Summary CSV for this sequence
            summary_df = pd.DataFrame(seq_summary_results)
            summary_df = summary_df.round(4)

            summary_path = os.path.join(OUTPUT_DIR, f"{seq}_benchmark_summary.csv")
            
            if os.path.exists(summary_path):
                try:
                    existing_df = pd.read_csv(summary_path)
                    # Filter out rows for trackers that are in the current run (overwrite logic)
                    current_trackers = summary_df['Tracker'].unique()
                    existing_df = existing_df[~existing_df['Tracker'].isin(current_trackers)]
                    # Combine old and new
                    summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
                    print(f"Merged with existing results in {summary_path}")
                except Exception as e:
                    print(f"Error reading existing summary: {e}. Overwriting.")

            summary_df = summary_df.sort_values(by=["Precision (CLE < 20px)", "Avg IoU"], ascending=False)

            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary for {seq} saved to: {summary_path}")

            # 2. Plots for this sequence
            if seq_frame_results:
                seq_full_df = pd.concat(seq_frame_results)
                
                # Plot IoU
                plt.figure(figsize=(10, 6))
                for tracker in seq_full_df['Tracker'].unique():
                    tracker_df = seq_full_df[seq_full_df['Tracker'] == tracker]
                    plt.plot(tracker_df['Frame'], tracker_df['IoU'], label=tracker, linewidth=1.5)
                
                plt.title(f'IoU over Time - {seq}')
                plt.xlabel('Frame')
                plt.ylabel('IoU')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.05)
                iou_plot_path = os.path.join(OUTPUT_DIR, f"{seq}_iou.png")
                plt.savefig(iou_plot_path)
                plt.close()
                print(f"Saved IoU plot to {iou_plot_path}")

                # Plot Center Error
                plt.figure(figsize=(10, 6))
                for tracker in seq_full_df['Tracker'].unique():
                    tracker_df = seq_full_df[seq_full_df['Tracker'] == tracker]
                    # Replace -1 with NaN for plotting so it doesn't skew the graph
                    errors = tracker_df['CenterError'].copy()
                    errors[errors == -1] = np.nan
                    plt.plot(tracker_df['Frame'], errors, label=tracker, linewidth=1.5)
                
                plt.title(f'Center Error over Time - {seq}')
                plt.xlabel('Frame')
                plt.ylabel('Center Error (px)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                error_plot_path = os.path.join(OUTPUT_DIR, f"{seq}_center_error.png")
                plt.savefig(error_plot_path)
                plt.close()
                print(f"Saved Center Error plot to {error_plot_path}")

    # --- SAVE GLOBAL DETAILED LOGS (Optional) ---
    if SAVE_DETAILED_LOGS and all_frame_results:
        full_df = pd.concat(all_frame_results)
        detailed_path = os.path.join(OUTPUT_DIR, "benchmark_full_frames.csv")
        full_df.to_csv(detailed_path, index=False)
        print(f"Detailed logs (all sequences) saved to: {detailed_path}")