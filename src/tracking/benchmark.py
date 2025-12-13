import cv2
import os
import sys
import numpy as np
import pandas as pd
import glob
import math
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
VISUALIZE = True 
# Save a big CSV with every frame from every tracker?
SAVE_DETAILED_LOGS = True 

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
    anno_file = os.path.join(ANNO_PATH, f"{seq_name}.txt")
    if not os.path.exists(anno_file):
        print(f"Missing annotation: {anno_file}")
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
    tracker = TRACKERS[tracker_name]()
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
    """Calculates SOT Metrics: Precision (CLE < 20px) and Success (IoU > 0.5)"""
    # Filter out occluded frames where Error is -1
    valid_df = df[df['CenterError'] != -1]
    
    summary = {
        "Tracker": df['Tracker'].iloc[0],
        "Sequence": df['Sequence'].iloc[0],
        "Avg FPS": df['FPS'].mean(),
        "Avg IoU": df['IoU'].mean(),
        "Precision (CLE < 20px)": (valid_df['CenterError'] < 20).mean(), # % of frames where error < 20px
        "Success Rate (IoU > 0.5)": (df['IoU'] > 0.5).mean()
    }
    return summary

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define which trackers and sequences to run
    trackers_to_test = ["BOOSTING", "MEDIANFLOW", "KCF", "CSRT", "MOSSE"] 
    # trackers_to_test = ["BOOSTING", "MEDIANFLOW", "MIL", "TLD"]
    
    sequences_to_test = ["person2"] # Add more here, e.g. ["car1", "person1"]
    
    all_frame_results = []
    summary_results = []

    for seq in sequences_to_test:
        for tracker_name in trackers_to_test:
            df = run_benchmark(seq, tracker_name)
            
            if df is not None and not df.empty:
                # 1. Collect Detailed Data
                all_frame_results.append(df)
                
                # 2. Calculate Summary for this run
                summ = calculate_summary(df)
                summary_results.append(summ)
                print(f"Finished {tracker_name} on {seq} -> Precision: {summ['Precision (CLE < 20px)']:.2%}")

    # --- SAVE OUTPUTS ---
    if summary_results:
        # 1. The Summary CSV (One row per tracker/sequence)
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        print(summary_df)

        # 2. The Detailed CSV and Plots
        if all_frame_results:
            full_df = pd.concat(all_frame_results)
            
            if SAVE_DETAILED_LOGS:
                detailed_path = os.path.join(OUTPUT_DIR, "benchmark_full_frames.csv")
                full_df.to_csv(detailed_path, index=False)
                print(f"Detailed logs saved to: {detailed_path}")

            # --- PLOTTING ---
            print("\nGenerating plots...")
            unique_sequences = full_df['Sequence'].unique()
            for seq in unique_sequences:
                seq_df = full_df[full_df['Sequence'] == seq]
                
                # Plot IoU
                plt.figure(figsize=(10, 6))
                for tracker in seq_df['Tracker'].unique():
                    tracker_df = seq_df[seq_df['Tracker'] == tracker]
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
                for tracker in seq_df['Tracker'].unique():
                    tracker_df = seq_df[seq_df['Tracker'] == tracker]
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