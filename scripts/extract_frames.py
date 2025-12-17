import cv2
import os
import argparse

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{count:04d}.jpg", frame)
        count += 1
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_folder)
