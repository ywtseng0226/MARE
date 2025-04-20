import os
import glob
import cv2
import numpy as np

# Define paths for prediction, input, and ground truth images
INPUT_DIR = "/home/ywt/research/25-AAAI-SSC/datasets/kitti360/data_2d_raw/2013_05_28_drive_0009_sync/image_2"
PRED_DIR = "/home/ywt/research/25-AAAI-SSC/ssc-ywt-mare/outputs/KITTI360/figures/2013_05_28_drive_0009_sync"
GT_DIR = "/home/ywt/research/25-AAAI-SSC/ssc-ywt-mare/outputs/KITTI360/figures/2013_05_28_drive_0009_sync"

OUTPUT_VIDEO = "/home/ywt/research/25-AAAI-SSC/ssc-ywt-mare/outputs/KITTI360/videos/2013_05_28_drive_0009_sync_demo.mp4"
FPS = 5  # Frames per second

def get_frame_id(filename):
    """Extract frame ID from the filename (excluding '_pred' or '_gt' suffix)."""
    return os.path.splitext(os.path.basename(filename))[0]  # e.g., "0001_pred"

def find_matched_frames(pred_dir, input_dir, gt_dir):
    """Find matching input and ground truth images for each prediction frame."""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*pred.png")))
    matched_frames = []

    for pred_file in pred_files:
        frame_id = get_frame_id(pred_file).replace("_pred", "")  # Extract base frame_id
        input_file = os.path.join(input_dir, f"{frame_id}.png")
        gt_file = os.path.join(gt_dir, f"{frame_id}_gt.png")

        if os.path.exists(input_file) and os.path.exists(gt_file):
            matched_frames.append((input_file, pred_file, gt_file))

    return matched_frames

def create_video(matched_frames, output_video, fps):
    """Create a video where:
       - Prediction and Ground Truth are merged **horizontally**.
       - The merged result is then **stacked vertically** with the Input image.
    """
    if not matched_frames:
        print("No matching frames found. Exiting.")
        return

    # Read the first image to determine dimensions
    first_input = cv2.imread(matched_frames[0][0])
    height, width, _ = first_input.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height * 2))

    # Process frames
    for input_file, pred_file, gt_file in matched_frames:
        img_input = cv2.imread(input_file)
        img_pred = cv2.imread(pred_file)
        img_gt = cv2.imread(gt_file)

        if img_input is None or img_pred is None or img_gt is None:
            print(f"Error loading images for frame {input_file}, skipping.")
            continue

        # Resize pred and gt images to match the input image dimensions
        img_pred = cv2.resize(img_pred, (width // 2, height))
        img_gt = cv2.resize(img_gt, (width // 2, height))

        # Merge Prediction and Ground Truth **horizontally**
        merged_pred_gt = np.hstack((img_pred, img_gt))

        # Stack the merged result **vertically** with the Input image
        stacked_image = np.vstack((img_input, merged_pred_gt))

        # Write frame to video
        video_writer.write(stacked_image)
        print(f"Added frame: {input_file}")

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    matched_frames = find_matched_frames(PRED_DIR, INPUT_DIR, GT_DIR)
    create_video(matched_frames, OUTPUT_VIDEO, FPS)
