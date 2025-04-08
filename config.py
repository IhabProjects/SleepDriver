"""
SleepDriver Configuration Module

This module contains all configurable parameters for the SleepDriver drowsiness detection system.
Adjust these settings to fine-tune the system for different users, environments, and hardware.

Note: After changing parameters, restart the application for changes to take effect.
"""

###################
# CORE PARAMETERS #
###################

# Eye Aspect Ratio (EAR) threshold for determining closed eyes
# Range: 0.15-0.25 (typical values)
# - Lower values (e.g., 0.18) = More sensitive detection (detects slightly closed eyes)
# - Higher values (e.g., 0.25) = Less sensitive (only detects fully closed eyes)
# Adjust based on individual facial features and lighting conditions
EYE_AR_THRESHOLD = 0.22

# Number of consecutive frames the eye must be below threshold to trigger alarm
# Range: 15-40 frames (at 30fps, this equals 0.5-1.3 seconds)
# - Lower values = Faster alerts but more false positives
# - Higher values = Fewer false positives but delayed alerts
# Adjust based on use case (lower for safety-critical applications)
EYE_AR_CONSEC_FRAMES = 25

#####################
# HARDWARE SETTINGS #
#####################

# Camera settings
CAMERA_INDEX = 0      # Camera device index (0=first camera, 1=second camera, etc.)
FRAME_WIDTH = 640     # Frame width in pixels (higher = more detail but slower processing)
FRAME_HEIGHT = 480    # Frame height in pixels (higher = more detail but slower processing)

####################
# MEDIAPIPE PARAMS #
####################

# MediaPipe Face Mesh detection parameters
# Note: These are set in the sleep_detector.py file:
# - max_num_faces=1: Focus on single user (driver)
# - refine_landmarks=True: Better accuracy for eye landmarks
# - min_detection_confidence=0.5: Balance between detection rate and false positives
# - min_tracking_confidence=0.5: Balance between tracking stability and adaptability

####################
# ALERT SETTINGS   #
####################

# Audio alert settings
ALARM_SOUND = "alarm.wav"    # Path to sound file (default=built-in beeping)
ALARM_VOLUME = 0.9           # Volume level (0.0=silent, 1.0=maximum)

####################
# UI SETTINGS      #
####################

# Visual elements configuration
SHOW_EAR_VALUE = True        # Display the EAR value on screen
SHOW_LANDMARKS = True        # Show facial landmarks visualization in debug mode

# Colors (in BGR format - OpenCV uses BGR instead of RGB)
TEXT_COLOR = (255, 0, 0)     # Alert text color (Blue=0, Green=0, Red=255) = Red
FRAME_COLOR = (0, 255, 0)    # Face frame color (Blue=0, Green=255, Red=0) = Green

####################
# DEBUG SETTINGS   #
####################

# Debugging options
SHOW_EYE_PROCESSING = True   # Enable detailed eye processing visualization

####################
# LOGGING SETTINGS #
####################

# Data logging for analysis
ENABLE_LOGGING = False                  # Enable/disable event logging
LOG_FILE = "sleep_detection_log.txt"    # Path to log file
