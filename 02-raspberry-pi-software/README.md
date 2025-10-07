# Raspberry Pi Software Stack

## Overview
This section covers the complete software ecosystem for Raspberry Pi, including operating system, development tools, programming environments, and libraries essential for building AI-powered vision systems.

## Table of Contents
1. [Operating System](#operating-system)
2. [Development Environment](#development-environment)
3. [Programming Languages](#programming-languages)
4. [Essential Libraries](#essential-libraries)
5. [System Configuration](#system-configuration)
6. [Performance Optimization](#performance-optimization)

## Operating System

### Raspberry Pi OS (Raspbian)
Raspberry Pi OS is the official operating system, based on Debian Linux.

#### Variants Available:
1. **Raspberry Pi OS with Desktop**: Full GUI environment
2. **Raspberry Pi OS Lite**: Minimal command-line installation
3. **Raspberry Pi OS (64-bit)**: Optimized for Pi 4's ARM64 architecture

#### Key Features:
- **Kernel**: Linux kernel optimized for ARM architecture
- **Package Manager**: APT (Advanced Package Tool)
- **Desktop Environment**: LXDE-based (Lightweight)
- **Pre-installed Software**: Python, Scratch, LibreOffice, Chromium

### Installation Process
```bash
# Download Raspberry Pi Imager
# Flash OS image to microSD card
# Boot and initial setup

# First boot configuration
sudo raspi-config

# Update system
sudo apt update && sudo apt upgrade -y
```

### Alternative Operating Systems
1. **Ubuntu Server/Desktop**: Full Ubuntu experience
2. **DietPi**: Lightweight distribution
3. **OpenELEC/LibreELEC**: Media center focused
4. **Arch Linux ARM**: Rolling release distribution

## Development Environment

### Integrated Development Environments (IDEs)

#### 1. Thonny Python IDE
```bash
# Pre-installed on Raspberry Pi OS
# Features:
# - Simple interface for beginners
# - Built-in debugger
# - Variable inspector
# - Step-through debugging
```

#### 2. Visual Studio Code
```bash
# Installation
sudo apt update
sudo apt install code

# Features:
# - IntelliSense
# - Git integration
# - Extensions marketplace
# - Remote development support
```

#### 3. PyCharm Community Edition
```bash
# Installation via snap
sudo apt install snapd
sudo snap install pycharm-community --classic

# Features:
# - Advanced Python development
# - Integrated debugger
# - Version control integration
# - Code analysis tools
```

### Version Control
```bash
# Git installation and setup
sudo apt install git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize repository
git init
git add .
git commit -m "Initial commit"
```

### Package Management
```bash
# System packages (APT)
sudo apt install package-name

# Python packages (pip)
pip3 install package-name

# Virtual environments
python3 -m venv venv_name
source venv_name/bin/activate
```

## Programming Languages

### Python (Primary Language)
Python is the preferred language for Raspberry Pi development.

#### Python 3 Features:
- **Version**: Python 3.9+ on latest Raspberry Pi OS
- **Package Manager**: pip3
- **Virtual Environments**: venv, virtualenv
- **Performance**: Adequate for most applications

#### Essential Python Packages:
```bash
# Core scientific computing
pip3 install numpy scipy matplotlib

# Computer vision
pip3 install opencv-python

# Machine learning
pip3 install tensorflow scikit-learn

# GPIO control
pip3 install RPi.GPIO gpiozero

# Camera interface
pip3 install picamera2

# Data handling
pip3 install pandas
```

### C/C++ (Performance Critical)
For performance-critical applications:

```bash
# Install build tools
sudo apt install build-essential cmake

# Example compilation
gcc -o program program.c -lwiringPi
g++ -o program program.cpp -std=c++17
```

### Shell Scripting (System Automation)
```bash
#!/bin/bash
# System automation scripts
# Service management
# Hardware configuration
```

## Essential Libraries

### GPIO Control Libraries

#### 1. RPi.GPIO
```python
import RPi.GPIO as GPIO
import time

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

# Control
GPIO.output(18, GPIO.HIGH)
time.sleep(1)
GPIO.output(18, GPIO.LOW)

# Cleanup
GPIO.cleanup()
```

#### 2. gpiozero (Recommended)
```python
from gpiozero import LED, Button
from signal import pause

led = LED(18)
button = Button(2)

button.when_pressed = led.on
button.when_released = led.off

pause()
```

### Camera Libraries

#### 1. picamera2 (New Interface)
```python
from picamera2 import Picamera2
import time

picam2 = Picamera2()
camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)
picam2.start()

time.sleep(2)
picam2.capture_file("image.jpg")
picam2.stop()
```

#### 2. OpenCV Integration
```python
import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### Computer Vision Libraries

#### OpenCV (Open Source Computer Vision)
```python
import cv2
import numpy as np

# Read image
img = cv2.imread('image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 50, 150)

# Display result
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Machine Learning Libraries

#### TensorFlow Lite
```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Make prediction
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```

## System Configuration

### raspi-config Tool
```bash
sudo raspi-config

# Key configuration options:
# 1. Enable Camera Interface
# 2. Enable SPI/I2C
# 3. GPU Memory Split
# 4. Overclock Settings
# 5. SSH Access
# 6. VNC Server
```

### GPU Memory Configuration
```bash
# Edit config.txt
sudo nano /boot/config.txt

# Add or modify:
gpu_mem=128  # Allocate 128MB to GPU (good for computer vision)
```

### Camera Configuration
```bash
# Enable camera in raspi-config or edit config.txt
echo 'start_x=1' >> /boot/config.txt
echo 'gpu_mem=128' >> /boot/config.txt

# Reboot required
sudo reboot
```

### SSH and Remote Access
```bash
# Enable SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# VNC Server (for remote desktop)
sudo apt install realvnc-vnc-server
sudo systemctl enable vncserver-x11-serviced
```

## Performance Optimization

### CPU Optimization
```bash
# Check current CPU frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Overclock settings in /boot/config.txt
arm_freq=1750          # CPU frequency
over_voltage=2         # Voltage adjustment
gpu_freq=600           # GPU frequency

# Temperature monitoring
vcgencmd measure_temp
```

### Memory Optimization
```bash
# Check memory usage
free -h
htop

# Reduce GPU memory for headless systems
gpu_mem=16  # Minimum for headless operation

# Swap configuration
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Storage Optimization
```bash
# Use faster storage
# - High-speed microSD cards (Class 10, UHS-I)
# - USB 3.0 drives for better performance

# Enable ZRAM (compressed RAM)
sudo apt install zram-tools
echo 'ALGO=lz4' | sudo tee -a /etc/default/zramswap
sudo service zramswap reload
```

### Python Performance
```python
# Use NumPy for numerical operations
import numpy as np

# Avoid Python loops for array operations
# Bad:
result = []
for i in range(len(array)):
    result.append(array[i] * 2)

# Good:
result = array * 2  # NumPy vectorized operation

# Use compiled libraries (OpenCV, NumPy, SciPy)
# Consider Cython for critical sections
```

## Service Management

### Systemd Services
```bash
# Create service file
sudo nano /etc/systemd/system/vision-system.service

# Service content:
[Unit]
Description=Vision System Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/vision-system
ExecStart=/usr/bin/python3 /home/pi/vision-system/main.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable vision-system.service
sudo systemctl start vision-system.service

# Check status
sudo systemctl status vision-system.service
```

### Log Management
```bash
# View system logs
journalctl -u vision-system.service

# Application logging in Python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Vision system started")
```

## Development Best Practices

### Code Organization
```
project/
├── main.py              # Entry point
├── config/
│   ├── __init__.py
│   └── settings.py      # Configuration management
├── src/
│   ├── __init__.py
│   ├── camera/          # Camera modules
│   ├── vision/          # Computer vision processing
│   └── hardware/        # GPIO and hardware control
├── tests/               # Unit tests
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

### Configuration Management
```python
# config/settings.py
import os
from dataclasses import dataclass

@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    framerate: int = 30

@dataclass
class VisionConfig:
    threshold: int = 100
    min_area: int = 1000

class Config:
    camera = CameraConfig()
    vision = VisionConfig()
    
    # Environment-based configuration
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
```

### Error Handling
```python
import logging
from contextlib import contextmanager

@contextmanager
def camera_context():
    """Ensure camera is properly initialized and cleaned up"""
    camera = None
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        camera.start()
        yield camera
    except Exception as e:
        logging.error(f"Camera error: {e}")
        raise
    finally:
        if camera:
            camera.stop()
```

## Practical Exercises

### Exercise 1: Environment Setup
1. Install Raspberry Pi OS
2. Configure SSH and VNC access
3. Set up Python virtual environment
4. Install essential packages

### Exercise 2: GPIO Programming
1. Control LED with button input
2. Read sensor data (temperature, humidity)
3. Implement PWM motor control

### Exercise 3: Camera Programming
1. Capture still images
2. Stream video feed
3. Basic image processing with OpenCV

### Exercise 4: System Integration
1. Create systemd service
2. Implement logging
3. Add configuration management
4. Error handling and recovery

## Troubleshooting Common Issues

### Camera Issues
```bash
# Check camera connection
vcgencmd get_camera

# Enable camera interface
sudo raspi-config
# Advanced Options → Camera → Enable

# Check for conflicts
lsmod | grep bcm2835
```

### Performance Issues
```bash
# Monitor system resources
htop
iotop
vcgencmd measure_temp

# Check for throttling
vcgencmd get_throttled
```

### Network Issues
```bash
# Check network connectivity
ping google.com
ip addr show

# Wi-Fi configuration
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

## References

1. [Raspberry Pi OS Documentation](https://www.raspberrypi.org/documentation/raspbian/)
2. [Python GPIO Library Documentation](https://gpiozero.readthedocs.io/)
3. [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/)
4. [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)

---

**Next Section**: [AI Accelerator →](../03-ai-accelerator/README.md)