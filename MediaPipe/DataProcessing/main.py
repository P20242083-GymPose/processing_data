import cv2
import os
import subprocess
import re

# Function to display video and flag rep boundaries manually with slow motion
def flag_reps_interactively(video_file, playback_factor=0.25):  # Factor to slow down (0.25 = 25% speed)
    cap = cv2.VideoCapture(video_file)
    rep_boundaries = []

    # Get the original frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int((1 / fps) * 1000 * (1 / playback_factor))  # Increase the delay for slower playback

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        key = cv2.waitKey(delay)  # Apply the delay to slow down the video

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)  # Get the current time in ms

        if key == ord('s'):  # Press 's' to mark start of the rep
            print(f"Start rep at {current_time} ms")
            rep_boundaries.append({"start": current_time})
        elif key == ord('e') and len(rep_boundaries) > 0:  # Press 'e' to mark end
            print(f"End rep at {current_time} ms")
            rep_boundaries[-1]["end"] = current_time

        if key == ord('q'):  # Press 'q' to quit and stop flagging
            break

    cap.release()
    cv2.destroyAllWindows()

    return rep_boundaries

# Function to get the next available file number for naming
def get_next_video_number(output_folder, prefix="squat_"):
    # List all files in the output folder
    files = os.listdir(output_folder)

    # Find all files that match the format squat_<number>.mp4
    numbers = [int(re.findall(rf'{prefix}(\d+)', f)[0]) for f in files if re.match(rf'{prefix}\d+\.mp4', f)]

    if numbers:
        # Get the highest number and add 1 to it
        return max(numbers) + 1
    else:
        # If no files exist, start with 1
        return 1

# Function to cut the video by flagged rep boundaries
def split_video_by_reps(video_file, rep_boundaries, output_folder, prefix="squat_"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the output folder if it doesn't exist

    # Get the next available number to start naming files
    next_number = get_next_video_number(output_folder, prefix)

    for i, rep in enumerate(rep_boundaries):
        start_time = rep['start'] / 1000  # Convert ms to seconds for FFmpeg
        end_time = rep['end'] / 1000
        output_file = f"{output_folder}/{prefix}{next_number}.mp4"
        
        ffmpeg_command = [
            'ffmpeg', '-i', video_file,
            '-ss', str(start_time), '-to', str(end_time),
            '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', output_file
        ]
        
        print(f"Cutting rep {i + 1}: {start_time}s to {end_time}s as {output_file}")
        subprocess.run(ffmpeg_command)

        next_number += 1  # Increment the number for the next file

    print("Reps saved successfully!")

# Main function to run the process
def main():
    video_file = 'squat_video.mp4'  # Path to your video file
    output_folder = '/home/dele/Documents/Thesis_Project/limb_recognition/MediaPipe/training_vids/good_posture/vids'  # Target output folder
    playback_factor = 0.50  # Set to 25% speed (four times slower)

    # Step 1: Flag reps interactively with slow motion
    rep_boundaries = flag_reps_interactively(video_file, playback_factor)

    # Step 2: Split and save videos based on flagged reps
    split_video_by_reps(video_file, rep_boundaries, output_folder)

# Run the main function
if __name__ == "__main__":
    main()
