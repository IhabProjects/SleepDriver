# SleepDriver

<div align="center">
  <img src="https://developers.google.com/static/mediapipe/images/solutions/face_landmarker_keypoints.png" alt="SleepDriver Logo" width="200"/>
  <h3>Real-time Drowsiness Detection System</h3>
  <p>Advanced computer vision technology to prevent accidents caused by driver fatigue</p>
</div>

---

## 📋 Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [System Flow](#system-flow)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

SleepDriver is a state-of-the-art drowsiness detection system that monitors eye state in real-time to detect signs of driver fatigue. When drowsiness is detected, the system triggers audible alerts to prevent accidents.

The system uses MediaPipe's Face Mesh technology to accurately track facial landmarks and calculate the Eye Aspect Ratio (EAR), providing reliable drowsiness detection even in challenging lighting conditions and varying head positions.

---

## ✨ Features

- **Real-time Face & Eye Tracking**: Uses MediaPipe's 468-point face mesh for precise landmark detection
- **Advanced Drowsiness Detection**: Calculates Eye Aspect Ratio (EAR) with temporal smoothing for reliable fatigue monitoring
- **Immediate Alerts**: Triggers audio alarms when drowsiness is detected
- **Customizable Sensitivity**: Easily adjust thresholds to match individual users and environments
- **Debug Visualization**: Real-time display of face mesh, eye landmarks, and metrics for system tuning
- **Event Logging**: Records drowsiness events for post-analysis and system improvement
- **Cross-platform**: Works on Windows, macOS, and Linux

---

## 🏗️ Technical Architecture

SleepDriver implements a multi-stage pipeline for drowsiness detection:

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌─────────────┐    ┌────────────┐
│  Video     │───▶│ MediaPipe  │───▶│  Eye       │───▶│ Drowsiness  │───▶│  Alert     │
│  Capture   │    │ Face Mesh  │    │  Analysis  │    │ Detection   │    │  System    │
└────────────┘    └────────────┘    └────────────┘    └─────────────┘    └────────────┘
```

### Core Components

1. **Video Capture**:
   - Reads frames from the camera
   - Applies pre-processing for optimal face detection

2. **Face Mesh Detection**:
   - Uses MediaPipe's Face Mesh model (468 landmarks)
   - Provides 3D face topology with sub-pixel precision

3. **Eye Analysis**:
   - Extracts eye-specific landmarks (16 points per eye)
   - Calculates EAR (Eye Aspect Ratio) for both eyes
   - Applies temporal smoothing to reduce noise

4. **Drowsiness Detection**:
   - Monitors EAR across consecutive frames
   - Applies thresholding with hysteresis for stability
   - Maintains state machine for drowsiness events

5. **Alert System**:
   - Generates audible alerts with PyGame
   - Visual indicators on the interface
   - Optional logging of events

---

## 🔄 System Flow

The following flowchart illustrates the system's operation:

```
┌─────────────┐
│  Start      │
└──────┬──────┘
       ▼
┌──────────────┐
│ Capture Frame│
└──────┬───────┘
       ▼
┌──────────────┐
│ Detect Face  │◄────┐
│ Landmarks    │     │
└──────┬───────┘     │
       │             │
       ▼             │
┌──────────────┐     │
│ Extract Eye  │     │
│ Landmarks    │     │
└──────┬───────┘     │
       │             │
       ▼             │
┌──────────────┐     │
│ Calculate    │     │
│ EAR Value    │     │
└──────┬───────┘     │
       │             │
       ▼             │
┌──────────────┐     │
│ Apply        │     │
│ Smoothing    │     │
└──────┬───────┘     │
       │             │
       ▼             │
┌──────────────┐     │
│ EAR <        │     │
│ Threshold?   │     │
└──────┬───────┘     │
       │             │
       ▼             │
┌──────────────┐     │
│ Increment    │     │
│ Counter      │     │
└──────┬───────┘     │
       │             │
       ▼             │
┌──────────────┐     │
│ Counter >    │ No  │
│ Frames?      ├─────┘
└──────┬───────┘
       │ Yes
       ▼
┌──────────────┐
│ Trigger      │
│ Alarm        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Next Frame   │
└──────────────┘
```

---

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Speakers for alert sounds

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sleepDriver.git
   cd sleepDriver
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the installation**:
   ```bash
   python test_installation.py
   ```

### Required Python Packages

| Package       | Version  | Purpose                                      |
|---------------|----------|----------------------------------------------|
| opencv-python | ≥4.5.0   | Computer vision and image processing         |
| numpy         | ≥1.20.0  | Numerical operations and array manipulation  |
| pygame        | ≥2.0.0   | Audio playback for alerts                    |
| mediapipe     | ≥0.8.10  | Face mesh landmark detection                 |
| matplotlib    | ≥3.4.0   | Optional visualization and debug plots       |

---

## 🚀 Usage

### Basic Usage

Run the system with default settings:

```bash
python sleep_detector.py
```

### Advanced Options

```bash
python sleep_detector.py --ear 0.15 --frames 25 --debug
```

### Command Line Arguments

| Argument    | Description                                 | Default Value              |
|-------------|---------------------------------------------|----------------------------|
| `--ear`     | Eye aspect ratio threshold                  | 0.17                       |
| `--frames`  | Consecutive frames to trigger alarm         | 20                         |
| `--camera`  | Camera index to use                         | 0 (first camera)           |
| `--log`     | Enable logging of drowsiness events         | False                      |
| `--debug`   | Show debug information and visualization    | False                      |
| `--silent`  | Run without sound alerts                    | False                      |

### Keyboard Controls

While the application is running:

| Key | Action                |
|-----|------------------------|
| `q` | Quit the application   |
| `d` | Toggle debug mode      |
| `r` | Reset counters         |

---

## ⚙️ Configuration

Adjust system parameters in `config.py`:

```python
# Eye detection thresholds
EYE_AR_THRESHOLD = 0.22       # Lower value = more sensitive
EYE_AR_CONSEC_FRAMES = 25     # Lower value = faster alert

# Camera settings
CAMERA_INDEX = 0              # Change for different cameras
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Alert settings
ALARM_VOLUME = 0.9            # Volume level (0.0 to 1.0)
```

### Fine-tuning for Different Users

The system may require adjustment for different users:

1. **EAR Threshold**: Lower for users with naturally narrower eyes
2. **Frame Threshold**: Lower for faster alerts, higher for fewer false positives
3. **Camera Position**: Best results when camera is at eye level

---

## 🧠 How It Works

### Eye Aspect Ratio (EAR)

The core algorithm relies on the Eye Aspect Ratio (EAR) formula:

<div align="center">
  <img src="https://learnopencv.com/wp-content/uploads/2021/06/eyes_landmarks.jpg" alt="EAR Formula" width="400"/>
</div>

In SleepDriver, we approximate this using MediaPipe landmarks:

```
EAR = eye_height / eye_width
```

Where:
- `eye_height` is the vertical distance between upper and lower eyelids
- `eye_width` is the horizontal distance between eye corners

The EAR value:
- **Open eyes**: Higher values (typically 0.2-0.3)
- **Closed eyes**: Lower values (typically <0.2)
- **Drowsy state**: Consistently low values over time

### MediaPipe Face Mesh

MediaPipe's Face Mesh provides a 3D mesh representation of the face with 468 landmarks:

<div align="center">
  <img src="https://developers.google.com/static/mediapipe/images/solutions/face_landmarker_keypoints.png" alt="Face Mesh" width="400"/>
</div>

Key advantages over traditional methods:
- Works in varying lighting conditions
- Handles different head poses
- Precise sub-pixel landmark locations
- Fast real-time performance

### Temporal Smoothing

To reduce false alarms, we employ:
1. EAR value smoothing using a 5-frame rolling average
2. Consecutive frame counting with hysteresis
3. Gradual counter reset for seamless operation

---

## 📊 Performance Considerations

### Hardware Requirements

| Component        | Minimum                  | Recommended               |
|------------------|--------------------------|---------------------------|
| CPU              | Dual-core 2.0 GHz        | Quad-core 2.5 GHz+        |
| RAM              | 4 GB                     | 8 GB+                     |
| Camera           | 640x480 @ 15 FPS         | 720p @ 30 FPS             |
| GPU              | Not required             | Integrated/dedicated      |

### Optimization Tips

- Reduce frame resolution for better performance on slower systems
- Adjust the tracking parameters in `config.py` for your specific camera
- Close other resource-intensive applications for better performance

---

## 🔧 Troubleshooting

### Common Issues

| Issue                           | Potential Solution                                     |
|---------------------------------|--------------------------------------------------------|
| Camera not detected             | Check camera index, permissions, and connections       |
| False alarms                    | Increase EAR and frame thresholds                      |
| No detection                    | Improve lighting, adjust camera position               |
| Poor performance                | Reduce resolution, close background applications       |
| No sound alerts                 | Check speaker connection, volume settings              |

### Debug Mode

Enable debug mode to visualize system performance:

```bash
python sleep_detector.py --debug
```

This provides real-time visualization of:
- Face mesh landmarks
- Eye contours
- EAR values
- Detection states and thresholds

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- Head pose estimation for improved drowsiness detection
- Machine learning model for personalized thresholds
- Mobile application port
- Integration with vehicle systems

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- MediaPipe team for their excellent face mesh implementation
- OpenCV community for computer vision tools
- PyGame for audio playback capabilities
- Contributors to the field of driver drowsiness detection research
