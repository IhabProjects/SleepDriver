#!/usr/bin/env python3
"""
SleepDriver - Advanced Drowsiness Detection System

This module implements real-time drowsiness detection using MediaPipe's Face Mesh model
to track facial landmarks and calculate the Eye Aspect Ratio (EAR). When drowsiness is
detected (eyes closed for an extended period), an alarm is triggered to alert the user.

Author: SleepDriver Team
Version: 1.0.0
License: MIT
"""

import cv2
import time
import argparse
import numpy as np
import pygame
import mediapipe as mp
from datetime import datetime
import os

# Import project modules
import config
from utils import play_alarm, log_drowsiness_event, resize_frame

# Initialize MediaPipe Face Mesh for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define eye landmarks indices based on MediaPipe Face Mesh topology
# These indices correspond to specific points on the face mesh model
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Left eye indices (16 points total: 8 for upper lid, 8 for lower lid)
LEFT_EYE_INDICES = [
    # Upper left eye points (outer to inner)
    362, 382, 381, 380, 374, 373, 390, 249,
    # Lower left eye points (outer to inner)
    263, 466, 388, 387, 386, 385, 384, 398
]

# Right eye indices (16 points total: 8 for upper lid, 8 for lower lid)
RIGHT_EYE_INDICES = [
    # Upper right eye points (outer to inner)
    33, 7, 163, 144, 145, 153, 154, 155,
    # Lower right eye points (outer to inner)
    133, 173, 157, 158, 159, 160, 161, 246
]

# Split indices for easier access to upper and lower eyelids
LEFT_EYE_UPPER = LEFT_EYE_INDICES[:8]
LEFT_EYE_LOWER = LEFT_EYE_INDICES[8:]
RIGHT_EYE_UPPER = RIGHT_EYE_INDICES[:8]
RIGHT_EYE_LOWER = RIGHT_EYE_INDICES[8:]

def parse_arguments():
    """
    Parse command line arguments for customizing the drowsiness detection system.

    Returns:
        argparse.Namespace: Parsed command-line arguments

    Command-line Options:
        --ear: Eye aspect ratio threshold (lower = more sensitive)
        --frames: Number of consecutive frames below threshold to trigger alarm
        --camera: Camera device index
        --log: Enable logging of drowsiness events
        --debug: Enable visualization of face mesh and metrics
        --silent: Disable audio alerts
    """
    parser = argparse.ArgumentParser(description='Advanced eye-based drowsiness detection system')
    parser.add_argument('--ear', type=float, default=0.17,
                        help='Eye aspect ratio threshold')
    parser.add_argument('--frames', type=int, default=20,
                        help='Number of consecutive frames to trigger alarm')
    parser.add_argument('--camera', type=int, default=config.CAMERA_INDEX,
                        help='Camera index (default is 0 for built-in webcam)')
    parser.add_argument('--log', action='store_true', default=config.ENABLE_LOGGING,
                        help='Enable logging of drowsiness events')
    parser.add_argument('--debug', action='store_true', default=config.SHOW_EYE_PROCESSING,
                        help='Show debug information for eye processing')
    parser.add_argument('--silent', action='store_true', default=False,
                        help='Run in silent mode (no alarm sound)')
    return parser.parse_args()

def calculate_ear(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) based on 3D landmarks from MediaPipe.

    The EAR measures the height-to-width ratio of the eye. When eyes are open,
    the EAR value is higher; when eyes close, the EAR value decreases. This metric
    is used to determine if a person's eyes are closed, indicating potential drowsiness.

    Formula: EAR = eye_height / eye_width

    Where:
    - eye_height: average vertical distance between upper and lower eyelids
    - eye_width: horizontal distance between eye corners

    Args:
        eye_landmarks (list): List of 16 landmarks for one eye (x, y, z coordinates)

    Returns:
        float: Calculated Eye Aspect Ratio value (typically between 0.15 and 0.35)
              Lower values indicate more closed eyes
    """
    # Ensure we have the right number of landmarks
    if len(eye_landmarks) != 16:
        return 0.0

    # Split into upper and lower eyelid landmarks
    upper_landmarks = eye_landmarks[:8]
    lower_landmarks = eye_landmarks[8:]

    # Calculate the mean y-value for upper and lower eyelid
    # This approach is more robust than using just a few points
    upper_y = np.mean([landmark.y for landmark in upper_landmarks])
    lower_y = np.mean([landmark.y for landmark in lower_landmarks])

    # Calculate the eye width (horizontal distance between eye corners)
    eye_width = abs(max([landmark.x for landmark in eye_landmarks]) -
                   min([landmark.x for landmark in eye_landmarks]))

    # Calculate vertical distance between eyelids (eye height)
    eye_height = abs(upper_y - lower_y)

    # Calculate EAR = eye_height / eye_width
    # Avoid division by zero
    if eye_width > 0:
        return eye_height / eye_width
    return 0.0

def extract_eye_landmarks(face_landmarks, eye_indices):
    """
    Extract specific eye landmarks from the complete face landmark set.

    Args:
        face_landmarks (MediaPipe.landmarks): Complete set of face landmarks (468 points)
        eye_indices (list): List of indices for the specific eye landmarks to extract

    Returns:
        list: Selected landmarks for the specified eye
    """
    return [face_landmarks.landmark[idx] for idx in eye_indices]

def main():
    """
    Main function implementing the drowsiness detection system workflow.

    Process:
    1. Initialize the system (camera, MediaPipe, arguments)
    2. Process video frames in real-time
    3. Detect face and extract eye landmarks
    4. Calculate EAR values and apply temporal smoothing
    5. Determine drowsiness state based on EAR thresholds
    6. Trigger alerts when drowsiness is detected
    7. Display results and debug information
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Initialize logger if enabled
    if args.log:
        from utils import initialize_logger
        initialize_logger(config.LOG_FILE)

    # Initialize pygame for sound alerts
    pygame.mixer.init()

    # Initialize MediaPipe Face Mesh with optimal parameters
    # - max_num_faces=1: Focus on the driver only
    # - refine_landmarks=True: Get more accurate eye landmarks
    # - min_detection_confidence: Threshold for face detection
    # - min_tracking_confidence: Threshold for landmark tracking
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize state variables for drowsiness detection
    COUNTER = 0          # Counter for consecutive frames below threshold
    ALARM_ON = False     # Flag to track if alarm is currently active

    # Initialize sliding window for EAR value smoothing (reduces false positives)
    ear_history = []

    # Initialize video capture from the specified camera
    print("[INFO] Starting video stream...")
    vs = cv2.VideoCapture(args.camera)

    # Set frame dimensions for consistent processing
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # Allow the camera sensor to warm up
    time.sleep(1.0)

    # Display control instructions
    print("[INFO] Press 'q' to quit the application")
    print("[INFO] Press 'd' to toggle debug mode")
    print("[INFO] Press 'r' to reset counters")

    # Debug mode flag (can be toggled during execution)
    debug_mode = args.debug

    # Main processing loop - process frames until user quits
    while True:
        # Grab a frame from the video stream
        ret, frame = vs.read()

        # Check if frame was successfully captured
        if not ret:
            print("[ERROR] Failed to grab frame - check your camera connection")
            break

        # Resize the frame to configured dimensions
        frame = resize_frame(frame, config.FRAME_WIDTH, config.FRAME_HEIGHT)

        # Convert BGR (OpenCV) to RGB (MediaPipe) color format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Mesh
        # Results contain detected face landmarks (if any)
        results = face_mesh.process(rgb_frame)

        # Default status - assume awake unless proven otherwise
        status_text = "Status: Awake"

        # Create a separate debug frame if in debug mode
        if debug_mode:
            debug_frame = frame.copy()

        # Check if face landmarks were detected
        if results.multi_face_landmarks:
            # Get the first face (driver's face)
            face_landmarks = results.multi_face_landmarks[0]

            # Draw face mesh visualization if in debug mode
            if debug_mode:
                # Draw full face mesh tesselation (triangles)
                mp_drawing.draw_landmarks(
                    image=debug_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Draw left eye contours for better visualization
                mp_drawing.draw_landmarks(
                    image=debug_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                # Draw right eye contours for better visualization
                mp_drawing.draw_landmarks(
                    image=debug_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            # Extract landmarks for both eyes
            left_eye_landmarks = extract_eye_landmarks(face_landmarks, LEFT_EYE_INDICES)
            right_eye_landmarks = extract_eye_landmarks(face_landmarks, RIGHT_EYE_INDICES)

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)

            # Average the EAR values from both eyes
            # This helps with robustness - if one eye is partially occluded or blinks
            ear = (left_ear + right_ear) / 2.0

            # Apply temporal smoothing: add to history and keep last 5 values
            # This reduces noise and prevents false positives from quick blinks
            ear_history.append(ear)
            if len(ear_history) > 5:
                ear_history.pop(0)  # Remove oldest value (sliding window)

            # Calculate smoothed EAR value (average of recent values)
            smoothed_ear = sum(ear_history) / len(ear_history)

            # Display the EAR value on the frame
            cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)

            # Drowsiness detection: check if EAR is below threshold
            if smoothed_ear < args.ear:
                # Increment frame counter for closed eyes
                COUNTER += 1

                # Display counter in debug mode
                if debug_mode:
                    cv2.putText(frame, f"Closed frames: {COUNTER}/{args.frames}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Check if eyes closed for sufficient consecutive frames
                if COUNTER >= args.frames:
                    # If alarm is not already on, trigger it
                    if not ALARM_ON:
                        ALARM_ON = True

                        # Play alarm sound unless silent mode is enabled
                        if not args.silent:
                            play_alarm(config.ALARM_SOUND, config.ALARM_VOLUME)

                        # Log the drowsiness event if logging is enabled
                        if args.log:
                            log_drowsiness_event(config.LOG_FILE)

                    # Update status text to indicate drowsiness
                    status_text = "Status: DROWSY!"

                    # Draw alert message on the frame
                    cv2.putText(frame, "WAKE UP!", (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)
            else:
                # Only reset counter if we're above the threshold by a good margin
                # This prevents flickering around the threshold value
                if smoothed_ear > (args.ear + 0.02):
                    # Gradual decrease to avoid rapid state changes
                    COUNTER = max(0, COUNTER - 1)

                    # Only turn off alarm after counter has been reset to zero
                    if COUNTER == 0:
                        ALARM_ON = False
        else:
            # No face detected - display message in debug mode
            if debug_mode:
                cv2.putText(frame, "No face detected", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Gradually decrease counter when no face is detected
            # This prevents immediate reset if face detection temporarily fails
            COUNTER = max(0, COUNTER - 1)

        # Display the current status on the frame
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)

        # Display timestamp on the frame for logging purposes
        timestamp = datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Display the main frame with drowsiness detection
        cv2.imshow("Sleep Detector (MediaPipe)", frame)

        # Display debug frame with mesh visualization if in debug mode
        if debug_mode:
            cv2.imshow("Debug View", debug_frame)

        # Process keyboard input for interactivity
        key = cv2.waitKey(1) & 0xFF

        # Break the loop on 'q' key press (quit)
        if key == ord("q"):
            break

        # Toggle debug mode on 'd' key press
        elif key == ord("d"):
            debug_mode = not debug_mode
            # Close debug window if debug mode is turned off
            if not debug_mode and 'debug_frame' in locals():
                cv2.destroyWindow("Debug View")

        # Reset statistics on 'r' key press
        elif key == ord("r"):
            COUNTER = 0
            ear_history = []
            print("[INFO] Counters reset")

    # Clean up resources when exiting
    vs.release()
    cv2.destroyAllWindows()
    print("[INFO] Sleep detection system stopped")

# Entry point of the application
if __name__ == "__main__":
    main()
