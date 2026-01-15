import os
import cv2
import glob
import numpy as np
import sys

# Configuration
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
SEQUENCES_TO_PROCESS = [
    "bike1", "bike3", "boat1", "boat2", "boat3", "car1", "car2", "car3", "car4",
    "car5", "car6", "car7", "car8", "car16", "car17", "car18", "person2", "person3", 
    "truck1", "truck2", "truck3", "wakeboard1", "wakeboard2", "wakeboard3"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data/datset", "UAV123")

# New Root for the rescaled dataset
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data/datset", "UAV123_640x480")

ANNO_PATH_UAV123 = os.path.join(DATA_ROOT, "anno", "UAV123")
ANNO_PATH_UAV20L = os.path.join(DATA_ROOT, "anno", "UAV20L")
SEQ_PATH = os.path.join(DATA_ROOT, "data_seq", "UAV123")

OUTPUT_ANNO_PATH = os.path.join(OUTPUT_ROOT, "anno", "UAV123")
OUTPUT_SEQ_PATH = os.path.join(OUTPUT_ROOT, "data_seq", "UAV123")

def process_sequence(seq_name):
    print(f"Processing sequence: {seq_name}")
    
    # 1. Locate Annotation File
    anno_file_src = os.path.join(ANNO_PATH_UAV123, f"{seq_name}.txt")
    if not os.path.exists(anno_file_src):
        anno_file_src = os.path.join(ANNO_PATH_UAV20L, f"{seq_name}.txt")
        if not os.path.exists(anno_file_src):
            print(f"  [Error] Annotation not found for {seq_name}")
            return

    # 2. Locate Image Folder
    img_folder_src = os.path.join(SEQ_PATH, seq_name)
    if not os.path.exists(img_folder_src):
        # Support for split naming convention if applicable
        img_folder_src = os.path.join(SEQ_PATH, seq_name.split('_')[0])
    
    if not os.path.exists(img_folder_src):
        print(f"  [Error] Image folder not found for {seq_name}")
        return

    images = sorted(glob.glob(os.path.join(img_folder_src, "*.jpg")))
    if not images:
        print(f"  [Error] No images found for {seq_name} in {img_folder_src}")
        return

    # 3. Determine Scaling Factors from first image
    first_img = cv2.imread(images[0])
    if first_img is None:
        print(f"  [Error] Could not read first image {images[0]}")
        return
        
    orig_h, orig_w = first_img.shape[:2]
    
    # Calculate crop to preserve aspect ratio
    target_ar = TARGET_WIDTH / TARGET_HEIGHT
    src_ar = orig_w / orig_h
    
    # Defaults (no crop)
    crop_x, crop_y = 0, 0
    crop_w, crop_h = orig_w, orig_h

    if src_ar > target_ar:
        # Source is wider: Crop width (sides) to match target AR
        new_w = int(orig_h * target_ar)
        crop_x = (orig_w - new_w) // 2
        crop_w = new_w
        print(f"  [Crop] Input is wider. Cropping sides. New size: {crop_w}x{crop_h} (x_offset={crop_x})")
    elif src_ar < target_ar:
        # Source is taller: Crop height (top/bottom)
        new_h = int(orig_w / target_ar)
        crop_y = (orig_h - new_h) // 2
        crop_h = new_h
        print(f"  [Crop] Input is taller. Cropping top/bottom. New size: {crop_w}x{crop_h} (y_offset={crop_y})")

    scale_x = TARGET_WIDTH / crop_w
    scale_y = TARGET_HEIGHT / crop_h
    
    print(f"  Original Size: {orig_w}x{orig_h} -> Crop: {crop_w}x{crop_h} -> Target: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"  Scale X: {scale_x:.4f}, Scale Y: {scale_y:.4f}")

    # 4. Process Images
    # Use the same folder name as found in source to maintain structure
    folder_name = os.path.basename(img_folder_src)
    output_seq_dir = os.path.join(OUTPUT_SEQ_PATH, folder_name)
    os.makedirs(output_seq_dir, exist_ok=True)

    print(f"  Resizing {len(images)} images...")
    for i, img_path in enumerate(images):
        img_name = os.path.basename(img_path)
        out_img_path = os.path.join(output_seq_dir, img_name)
        
        # Determine if we need to read the file (optimization: if exists, skip? No, explicit overwrite requested implicitly)
        img = cv2.imread(img_path)
        if img is None: 
            continue
        
        cropped_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        resized_img = cv2.resize(cropped_img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_img_path, resized_img)
        
        if i % 100 == 0:
            sys.stdout.write(f"\r  Progress: {i}/{len(images)}")
            sys.stdout.flush()
    print("")

    # 5. Process Annotations
    with open(anno_file_src, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        # Normalize delimiters (comma or space)
        clean_line = line.replace(',', ' ').strip()
        parts = clean_line.split()
        
        if not parts:
            new_lines.append("\n")
            continue
            
        try:
            # Parse box: x, y, w, h
            # Handle potential NaNs which might be strings in the file or just invalid floats
            box = [float(p) for p in parts]
            
            if np.isnan(box).any():
                new_line = "NaN,NaN,NaN,NaN\n" # Preserve structure for NaNs
            else:
                 # Adjust for Crop (Shift)
                bx_shifted = box[0] - crop_x
                by_shifted = box[1] - crop_y
                bw = box[2]
                bh = box[3]
                
                # Clip to visible area to ensure ground truth matches visual content
                x1 = max(0, bx_shifted)
                y1 = max(0, by_shifted)
                x2 = min(crop_w, bx_shifted + bw)
                y2 = min(crop_h, by_shifted + bh)
                
                vis_w = x2 - x1
                vis_h = y2 - y1
                
                if vis_w <= 0 or vis_h <= 0:
                    # Object is fully cropped out
                    new_line = "NaN,NaN,NaN,NaN\n"
                else:
                    # Scale to new resolution
                    new_box = [
                        x1 * scale_x,
                        y1 * scale_y,
                        vis_w * scale_x,
                        vis_h * scale_y
                    ]
                    # Write as comma-separated values
                    new_line = ",".join([f"{v:.2f}" for v in new_box]) + "\n"
            
            new_lines.append(new_line)
            
        except ValueError:
            # If line is not numeric, write it as is (headers, different formats)
            new_lines.append(line)

    os.makedirs(OUTPUT_ANNO_PATH, exist_ok=True)
    anno_out_file = os.path.join(OUTPUT_ANNO_PATH, f"{seq_name}.txt")
    
    with open(anno_out_file, 'w') as f:
        f.writelines(new_lines)
    print(f"  Saved rescaled annotations to {anno_out_file}")

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found at {DATA_ROOT}")
        exit(1)
        
    print(f"Starting dataset rescaling to {TARGET_WIDTH}x{TARGET_HEIGHT}...")
    for seq in SEQUENCES_TO_PROCESS:
        process_sequence(seq)
    print("\nDataset rescaling complete.")
