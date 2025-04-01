import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose

# Specify connections to visualize (ignore face points)
connections = [
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
]

# Define the header for the CSV file
header = ['Frame']
for connection in connections:
    header.append(f"{connection[0].name}_to_{connection[1].name}_start_x")
    header.append(f"{connection[0].name}_to_{connection[1].name}_start_y")
    header.append(f"{connection[0].name}_to_{connection[1].name}_end_x")
    header.append(f"{connection[0].name}_to_{connection[1].name}_end_y")

# Folder paths
video_folder_path = 'training_vids/good_posture/vids/'
csv_folder_path = 'training_vids/good_posture/csv/'

# Ensure the CSV folder exists
os.makedirs(csv_folder_path, exist_ok=True)

# Function to process a single video file
def process_video(video_path, csv_path, flip=False):
    # Start video capture from file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Initialize the CSV file to save the data
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)  # Write the header row
        
        # Initialize the Pose module
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip the frame if needed
                if flip:
                    frame = cv2.flip(frame, 1)  # Flip horizontally

                # Convert the image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image and find pose landmarks
                results = pose.process(image_rgb)

                # If landmarks are detected
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Prepare a row to store the coordinates
                    row = [frame_count]

                    # Draw and store the coordinates for each specified connection
                    for connection in connections:
                        start_landmark = landmarks[connection[0].value]
                        end_landmark = landmarks[connection[1].value]

                        # Get the coordinates
                        start_x = int(start_landmark.x * frame.shape[1])
                        start_y = int(start_landmark.y * frame.shape[0])
                        end_x = int(end_landmark.x * frame.shape[1])
                        end_y = int(end_landmark.y * frame.shape[0])

                        # Append the coordinates to the row
                        row.extend([start_x, start_y, end_x, end_y])

                    # Write the row to the CSV file
                    csvwriter.writerow(row)

                frame_count += 1

    # Release the video capture
    cap.release()

# Process each video in the folder
for filename in os.listdir(video_folder_path):
    if filename.endswith('.mp4'):
        video_path = os.path.join(video_folder_path, filename)
        
        # Define paths for the original and flipped CSV files
        csv_path_original = os.path.join(csv_folder_path, f"{os.path.splitext(filename)[0]}.csv")
        csv_path_flipped = os.path.join(csv_folder_path, f"{os.path.splitext(filename)[0]}_flipped.csv")

        # Process the original video
        process_video(video_path, csv_path_original, flip=False)

        # Process the flipped video
        process_video(video_path, csv_path_flipped, flip=True)

print("Processing complete. CSV files saved for both original and flipped videos.")
