import cv2
import os
import argparse

def save_frames(video_path, output_folder):
    # Create output directory if not existing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Accept video input
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file") 
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % fps == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count//fps:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Attempting to save: {frame_path}")
            if os.path.exists(frame_path):
                print(f"Saved {frame_path}")
            else:
                print(f"Failed to save {frame_path}")
            
        frame_count += 1
        
    # Release video capture and close all frames
    cap.release()
    cv2.destroyAllWindows()
    print("Completed extracting frames.")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video file.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('output_folder', type=str, help='Path to output folder for frames')
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' does not exist.")
        return
    
    save_frames(args.video_path, args.output_folder)

if __name__ == "__main__":
    main()
