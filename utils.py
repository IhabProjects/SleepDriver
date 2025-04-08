"""
SleepDriver Utility Functions Module

This module provides helper functions for the SleepDriver drowsiness detection system,
including audio alerts, image processing, logging, and various analysis algorithms.

The functions in this module handle common tasks needed by the main application logic,
keeping the core codebase clean and modular.
"""

import numpy as np
from scipy.spatial import distance as dist
import cv2
import pygame
import time
import logging
from datetime import datetime
import os

def initialize_logger(log_file):
    """
    Set up the application logger for event tracking and analysis.

    Configures a Python logger to record important events like drowsiness detection
    with timestamps for later review and analysis.

    Args:
        log_file (str): Path to the log file where events will be recorded
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("SleepDriver session started")

def play_alarm(sound_file=None, volume=1.0):
    """
    Play an audio alarm to alert the user when drowsiness is detected.

    This function handles both file-based sound playback and direct generation
    of alert sounds when no sound file is provided. It includes error handling
    to ensure some form of alert is produced even if audio playback fails.

    Args:
        sound_file (str, optional): Path to a sound file for the alarm.
                                   If None or file not found, generates a beep.
        volume (float): Volume level from 0.0 (silent) to 1.0 (maximum volume)
                       Default: 1.0
    """
    try:
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init(44100, -16, 2, 1024)

        # Set volume
        pygame.mixer.music.set_volume(volume)

        # Generate a simple beep if no sound file or file doesn't exist
        if sound_file is None or not os.path.exists(sound_file):
            # Create a sound buffer with a loud beep
            duration = 1.0  # seconds
            frequency = 880  # Hz (higher pitch for better alerting)
            sample_rate = 44100
            samples = int(duration * sample_rate)

            # Create a sound array with alternating frequencies for more attention
            # Two-tone alarm is more effective at capturing attention
            t = np.linspace(0, duration, samples, False)
            wave1 = np.sin(2 * np.pi * frequency * t) * 32767 * 0.7  # Primary tone
            wave2 = np.sin(2 * np.pi * (frequency*1.5) * t) * 32767 * 0.3  # Secondary tone
            tone = wave1 + wave2

            # Create stereo sound buffer
            buffer = np.zeros((samples, 2), dtype=np.int16)
            buffer[:, 0] = tone  # Left channel
            buffer[:, 1] = tone  # Right channel

            # Create and play sound
            sound = pygame.sndarray.make_sound(buffer)
            sound.play()
            print("[INFO] Alarm sound playing (generated beep)")
        else:
            # Play the sound file if it exists
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            print(f"[INFO] Alarm sound playing from file: {sound_file}")

    except Exception as e:
        print(f"[ERROR] Failed to play alarm: {e}")
        # Fallback to console bell (ASCII BEL character)
        # This provides a basic alert even if sound playback fails
        print("\a" * 3)  # Console bell

def calculate_eye_aspect_ratio(eye1, eye2):
    """
    Calculate an approximation of the Eye Aspect Ratio (EAR) using bounding boxes.

    Unlike the standard EAR calculation that requires precise eye landmarks,
    this function works with simpler eye bounding boxes. It calculates a ratio
    based on the height-to-width proportion of each eye and applies adjustments
    based on eye size to improve accuracy.

    Algorithm:
    1. Calculate height/width ratio for each eye
    2. Account for eye size (larger eyes = likely more open)
    3. Average and normalize to standard EAR range

    Args:
        eye1 (tuple): First eye coordinates (x, y, w, h)
        eye2 (tuple): Second eye coordinates (x, y, w, h)

    Returns:
        float: The estimated eye aspect ratio, typically in range 0.15-0.35
               Lower values indicate more closed eyes
    """
    # Unpack eye coordinates
    x1, y1, w1, h1 = eye1
    x2, y2, w2, h2 = eye2

    # Calculate aspect ratio for each eye (height/width)
    ratio1 = h1 / max(w1, 1)  # Avoid division by zero
    ratio2 = h2 / max(w2, 1)

    # Calculate areas (larger eyes generally mean more open)
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate average aspect ratio from both eyes
    # This improves robustness to partial occlusion of one eye
    avg_ratio = (ratio1 + ratio2) / 2.0

    # Use height-to-width ratio as the primary indicator
    # Multiply by scaling factor to map to standard EAR range
    ear = 0.27 * avg_ratio

    # Apply size-based adjustment
    # Larger eyes (by area) are more likely to be open eyes
    # This helps differentiate between naturally narrow eyes and closed eyes
    size_factor = min(0.03, (area1 + area2) / 20000)
    ear += size_factor

    # Clamp values to a reasonable EAR range (0.15-0.35)
    # This prevents extreme values that could confuse the drowsiness detection
    return min(max(ear, 0.15), 0.35)

def is_looking_away(eyes, frame_height, frame_width):
    """
    Determine if the user is looking away from the camera/road.

    This function analyzes the positions of detected eyes relative to the frame
    to determine if the person is looking off to the side or up/down instead of
    straight ahead. This can be used as an additional indicator of distraction
    or inattentiveness beyond just eye closure.

    Args:
        eyes (list): List of eye coordinates [(x, y, w, h), ...]
        frame_height (int): Height of the video frame in pixels
        frame_width (int): Width of the video frame in pixels

    Returns:
        bool: True if the person appears to be looking away, False otherwise
    """
    # Need at least two eyes to make a determination
    if len(eyes) < 2:
        return False

    # Calculate eye centers
    eye_centers = []
    for x, y, w, h in eyes[:2]:  # Use the first two eyes
        center_x = x + w/2
        center_y = y + h/2
        eye_centers.append((center_x, center_y))

    # Frame center coordinates
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    # Check if eyes are positioned too far to the edges
    # This indicates the person is looking to the side
    for cx, cy in eye_centers:
        # Horizontal check - if eyes are far to the sides
        # Eyes should be within the central 60% of the frame
        if cx < frame_width * 0.2 or cx > frame_width * 0.8:
            return True

        # Vertical check - if eyes are too high or low
        # Eyes should be in the upper half of the frame, but not too high
        if cy < frame_height * 0.15 or cy > frame_height * 0.6:
            return True

    # Eyes are in the expected position for forward gaze
    return False

def detect_eye_closure(eye_roi_gray):
    """
    Analyze the eye region to detect if an eye is closed using intensity histogram.

    This is a supplementary method to EAR-based detection that analyzes the
    pixel intensity distribution in the eye region. Closed eyes typically have
    more dark pixels (eyelashes, shadows) than open eyes.

    Algorithm:
    1. Enhance contrast with histogram equalization
    2. Calculate intensity histogram
    3. Weight darker pixels more heavily (closed eyes have more dark regions)
    4. Calculate weighted score

    Args:
        eye_roi_gray (numpy.ndarray): Grayscale image of the eye region

    Returns:
        float: Closure score between 0-1, where higher values indicate
               a higher probability that the eye is closed
    """
    # Apply histogram equalization to enhance contrast
    eye_roi_eq = cv2.equalizeHist(eye_roi_gray)

    # Calculate intensity histogram (256 bins)
    hist = cv2.calcHist([eye_roi_eq], [0], None, [256], [0, 256])

    # Normalize histogram to sum to 1 (probability distribution)
    hist = hist / hist.sum()

    # Calculate weighted sum - higher weights for darker pixels
    # Closed eyes have more dark pixels (eyelashes, shadows)
    weights = np.linspace(1.0, 0.1, 256)  # Higher weights for darker values
    closure_score = np.sum(hist.flatten() * weights)

    return closure_score

def log_drowsiness_event(log_file):
    """
    Record a drowsiness detection event to the log file.

    This function is called whenever drowsiness is detected to maintain
    a record of all incidents for later review and analysis.

    Args:
        log_file (str): Path to the log file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - Drowsiness detected\n")
        logging.info("Drowsiness event logged")
    except Exception as e:
        print(f"[ERROR] Failed to log drowsiness event: {e}")

def resize_frame(frame, width, height):
    """
    Resize a video frame to specified dimensions while maintaining aspect ratio.

    Consistent frame sizing ensures predictable processing times and
    helps normalize detection parameters across different cameras.

    Args:
        frame (numpy.ndarray): Input video frame
        width (int): Target width in pixels
        height (int): Target height in pixels

    Returns:
        numpy.ndarray: Resized frame
    """
    # Use INTER_AREA interpolation for downsizing (better quality)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
