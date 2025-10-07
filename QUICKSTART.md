# Quick Start Guide

## Project Setup

### 1. Hardware Requirements
- Raspberry Pi 4 (4GB+ recommended)
- MicroSD card (32GB+ Class 10)
- Raspberry Pi Camera Module v2 or v3
- Breadboard and jumper wires
- LEDs and resistors (220Ω)
- Push buttons
- 24V power supply (for industrial sensors - optional)

### 2. Software Installation

#### Install Raspberry Pi OS
1. Download Raspberry Pi Imager
2. Flash Raspberry Pi OS (64-bit recommended)
3. Enable SSH and camera in raspi-config

#### Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system packages
sudo apt install -y python3-pip python3-venv git
sudo apt install -y python3-opencv libatlas-base-dev
sudo apt install -y python3-picamera2

# Clone project (if from git repository)
git clone <repository-url>
cd miniproject

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Basic Hardware Setup

#### GPIO Connections (for demos)
```
Raspberry Pi GPIO Connections:
- GPIO 2  (Pin 3)  → Push button (with pull-up)
- GPIO 3  (Pin 5)  → Push button (with pull-up)
- GPIO 4  (Pin 7)  → Sensor input (with pull-up)
- GPIO 18 (Pin 12) → LED (with 220Ω resistor)
- GPIO 19 (Pin 35) → LED (with 220Ω resistor)
- GPIO 20 (Pin 38) → LED (with 220Ω resistor)
- GPIO 21 (Pin 40) → LED (with 220Ω resistor)
- GND              → Common ground for all components
```

#### Camera Connection
1. Connect camera ribbon cable to camera port
2. Enable camera interface: `sudo raspi-config` → Interface → Camera → Enable
3. Reboot: `sudo reboot`

### 4. Running Examples

#### Basic Camera Test
```bash
cd examples/basic_demos
python3 camera_test.py
```

#### GPIO Demo
```bash
python3 gpio_demo.py
```

#### Vision System Demo
```bash
python3 simple_vision_demo.py
```

### 5. Project Structure Navigation

#### Learning Path
1. **Hardware Understanding**: Start with `01-raspberry-pi-structure/`
2. **Software Setup**: Continue with `02-raspberry-pi-software/`
3. **AI Optimization**: Explore `03-ai-accelerator/`
4. **Vision Implementation**: Study `04-vision-system/`
5. **Electrical Integration**: Complete with `05-electrical-wiring/`

#### Practical Implementation
1. Run basic demos in `examples/basic_demos/`
2. Study source code in `src/`
3. Follow documentation in each numbered folder
4. Build complete system using all components

### 6. Troubleshooting

#### Camera Issues
```bash
# Check camera detection
vcgencmd get_camera

# If camera not detected:
sudo raspi-config  # Enable camera interface
sudo reboot
```

#### GPIO Issues
```bash
# Check GPIO permissions
sudo usermod -a -G gpio $USER
# Logout and login again
```

#### Package Installation Issues
```bash
# If OpenCV installation fails:
sudo apt install -y python3-opencv

# If picamera2 not available:
sudo apt install -y python3-picamera2

# For TensorFlow Lite on older Pi models:
pip install tflite-runtime
```

### 7. Next Steps

1. Complete the basic demos
2. Study each section's README thoroughly
3. Implement your own vision inspection task
4. Integrate with real industrial sensors
5. Add PLC communication capabilities
6. Deploy to production environment

### 8. Safety Notes

- Always power down before making connections
- Use appropriate voltage levels (3.3V for GPIO)
- Follow electrical safety guidelines in section 5
- Use proper grounding techniques
- Consider isolation for industrial environments

## Quick Demo Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Test camera functionality
python3 examples/basic_demos/camera_test.py

# Test GPIO operations
python3 examples/basic_demos/gpio_demo.py

# Run vision system demo
python3 examples/basic_demos/simple_vision_demo.py

# Create your own inspection application
# Follow the examples and adapt for your specific needs
```

For detailed information, refer to the documentation in each numbered folder.