# AI Accelerator for Raspberry Pi

## Overview
This section explores AI acceleration techniques and hardware solutions for enhancing machine learning performance on Raspberry Pi. We'll cover both software optimization and hardware acceleration options suitable for industrial vision systems.

## Table of Contents
1. [AI Acceleration Fundamentals](#ai-acceleration-fundamentals)
2. [Software Optimization](#software-optimization)
3. [Hardware Accelerators](#hardware-accelerators)
4. [TensorFlow Lite Optimization](#tensorflow-lite-optimization)
5. [Neural Processing Units](#neural-processing-units)
6. [Performance Benchmarking](#performance-benchmarking)

## AI Acceleration Fundamentals

### Why AI Acceleration is Needed
Raspberry Pi's ARM CPU, while capable, has limitations for AI workloads:
- **Limited FLOPS**: ~25 GFLOPS peak performance
- **Memory Bandwidth**: Shared between CPU and GPU
- **Power Constraints**: Thermal throttling under sustained load
- **Real-time Requirements**: Industrial applications need consistent performance

### Types of AI Acceleration

#### 1. Software Optimization
- Model quantization (FP32 → INT8)
- Model pruning and compression
- Optimized runtime libraries
- Efficient algorithms

#### 2. Hardware Acceleration
- GPU acceleration (VideoCore)
- Neural Processing Units (NPUs)
- Edge AI accelerators
- FPGA solutions

#### 3. Model Architecture Optimization
- MobileNet architectures
- EfficientNet models
- Knowledge distillation
- Neural Architecture Search (NAS)

## Software Optimization

### TensorFlow Lite Runtime
TensorFlow Lite provides optimized inference for edge devices.

#### Installation
```bash
# Install TensorFlow Lite runtime (lighter than full TensorFlow)
pip3 install tflite-runtime

# Alternative: Full TensorFlow (if needed)
pip3 install tensorflow
```

#### Basic TFLite Usage
```python
import tflite_runtime.interpreter as tflite
import numpy as np

class TFLiteModel:
    def __init__(self, model_path):
        """Initialize TensorFlow Lite interpreter"""
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Extract input shape
        self.input_shape = self.input_details[0]['shape']
    
    def predict(self, input_data):
        """Run inference on input data"""
        # Ensure input data matches expected shape and type
        input_data = np.array(input_data, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0]

# Usage example
model = TFLiteModel('model.tflite')
result = model.predict(preprocessed_image)
```

### Model Quantization
Convert models from 32-bit floating point to 8-bit integers.

#### Post-Training Quantization
```python
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('original_model.h5')

# Convert to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Representative dataset for better quantization
def representative_dataset():
    for _ in range(100):
        # Use real data samples
        data = np.random.random((1, 224, 224, 3)).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert model
quantized_model = converter.convert()

# Save quantized model
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_model)
```

#### Performance Comparison
```python
import time
import numpy as np

def benchmark_model(model_path, input_shape, num_runs=100):
    """Benchmark model inference time"""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warm-up runs
    dummy_input = np.random.random(input_shape).astype(np.float32)
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark runs
    start_time = time.time()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    return avg_time, fps

# Compare original vs quantized
original_time, original_fps = benchmark_model('original_model.tflite', (1, 224, 224, 3))
quantized_time, quantized_fps = benchmark_model('quantized_model.tflite', (1, 224, 224, 3))

print(f"Original: {original_time:.3f}s ({original_fps:.1f} FPS)")
print(f"Quantized: {quantized_time:.3f}s ({quantized_fps:.1f} FPS)")
print(f"Speedup: {original_time/quantized_time:.2f}x")
```

### OpenCV DNN Module
OpenCV's DNN module provides optimized inference for various frameworks.

```python
import cv2
import numpy as np

class OpenCVDNN:
    def __init__(self, model_path, config_path=None):
        """Initialize OpenCV DNN model"""
        if model_path.endswith('.tflite'):
            self.net = cv2.dnn.readNetFromTensorflow(model_path)
        elif model_path.endswith('.onnx'):
            self.net = cv2.dnn.readNetFromONNX(model_path)
        else:
            raise ValueError("Unsupported model format")
    
    def predict(self, image):
        """Run inference using OpenCV DNN"""
        blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (224, 224), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        return output

# Usage
dnn_model = OpenCVDNN('model.onnx')
result = dnn_model.predict(input_image)
```

## Hardware Accelerators

### GPU Acceleration (VideoCore VI)
The Raspberry Pi 4's GPU can accelerate certain AI operations.

#### OpenGL ES Compute Shaders
```python
# Limited support for compute operations
# Mainly useful for image preprocessing and postprocessing

import cv2

# Use GPU-accelerated OpenCV operations when available
# cv2.setUseOptimized(True)
# cv2.setNumThreads(4)  # Utilize all CPU cores
```

### External AI Accelerators

#### 1. Google Coral USB Accelerator
Edge TPU accelerator for TensorFlow Lite models.

```bash
# Installation
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
pip3 install pycoral
```

```python
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import numpy as np

class CoralAccelerator:
    def __init__(self, model_path):
        """Initialize Coral Edge TPU"""
        self.interpreter = edgetpu.make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        
    def predict(self, image):
        """Run inference on Edge TPU"""
        # Resize and normalize image
        input_tensor = common.set_resized_input(
            self.interpreter, image.shape[:2], 
            lambda size: cv2.resize(image, size)
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get results
        return classify.get_classes(self.interpreter)

# Usage
coral = CoralAccelerator('model_edgetpu.tflite')
results = coral.predict(input_image)
```

#### 2. Intel Movidius Neural Compute Stick
```bash
# Installation (OpenVINO toolkit)
pip3 install openvino-dev
```

```python
from openvino.runtime import Core

class MovidiusAccelerator:
    def __init__(self, model_path):
        """Initialize Intel Movidius NCS"""
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "MYRIAD")
        self.infer_request = self.compiled_model.create_infer_request()
        
    def predict(self, input_data):
        """Run inference on NCS"""
        self.infer_request.infer({0: input_data})
        return self.infer_request.get_output_tensor().data

# Usage
ncs = MovidiusAccelerator('model.xml')
result = ncs.predict(preprocessed_input)
```

## TensorFlow Lite Optimization

### Delegate APIs
TensorFlow Lite supports hardware-specific delegates for acceleration.

#### GPU Delegate (Limited on Pi)
```python
import tflite_runtime.interpreter as tflite

# GPU delegate (limited support on Raspberry Pi)
try:
    # This may not work on all Pi models
    interpreter = tflite.Interpreter(
        model_path="model.tflite",
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
except:
    # Fallback to CPU
    interpreter = tflite.Interpreter(model_path="model.tflite")
```

#### Custom Operations
```python
# Register custom operations for specialized hardware
def register_custom_ops():
    """Register custom TensorFlow Lite operations"""
    pass  # Implementation depends on specific hardware
```

### Model Architecture Optimization

#### MobileNet for Raspberry Pi
```python
import tensorflow as tf

def create_mobilenet_model(input_shape, num_classes, alpha=1.0):
    """Create optimized MobileNet model for Raspberry Pi"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,  # Width multiplier (0.35, 0.5, 0.75, 1.0)
        include_top=False,
        weights='imagenet'
    )
    
    # Add custom classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Example: Small model for real-time inference
model = create_mobilenet_model((224, 224, 3), 10, alpha=0.35)
```

#### EfficientNet Optimization
```python
def create_efficient_model(input_shape, num_classes):
    """Create EfficientNet model optimized for edge devices"""
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze early layers
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

## Neural Processing Units

### Raspberry Pi AI Kit (Future)
Raspberry Pi Foundation is developing AI acceleration solutions.

```python
# Placeholder for future AI kit integration
class RaspberryPiAI:
    def __init__(self):
        """Initialize Raspberry Pi AI accelerator"""
        pass
    
    def load_model(self, model_path):
        """Load model onto AI accelerator"""
        pass
    
    def predict(self, input_data):
        """Run inference on AI hardware"""
        pass
```

### Third-Party NPU Solutions

#### Hailo-8 AI Processor
```python
# Example integration (requires Hailo SDK)
class HailoAccelerator:
    def __init__(self, model_path):
        """Initialize Hailo AI processor"""
        # Implementation depends on Hailo SDK
        pass
    
    def predict(self, input_data):
        """Run inference on Hailo NPU"""
        # High-performance inference
        pass
```

## Performance Benchmarking

### Comprehensive Benchmark Suite
```python
import time
import psutil
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class BenchmarkResult:
    model_name: str
    avg_inference_time: float
    fps: float
    cpu_usage: float
    memory_usage: float
    temperature: float

class AIBenchmark:
    def __init__(self):
        """Initialize AI performance benchmark"""
        self.results: List[BenchmarkResult] = []
    
    def benchmark_model(self, model_path, input_shape, num_runs=100):
        """Comprehensive model benchmarking"""
        # Load model
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Warm-up
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # Monitor system resources
        cpu_percent = []
        memory_percent = []
        
        # Benchmark with resource monitoring
        start_time = time.time()
        for i in range(num_runs):
            if i % 10 == 0:  # Sample every 10 runs
                cpu_percent.append(psutil.cpu_percent())
                memory_percent.append(psutil.virtual_memory().percent)
            
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        # Get system temperature
        temp = self.get_temperature()
        
        result = BenchmarkResult(
            model_name=model_path,
            avg_inference_time=avg_time,
            fps=fps,
            cpu_usage=np.mean(cpu_percent),
            memory_usage=np.mean(memory_percent),
            temperature=temp
        )
        
        self.results.append(result)
        return result
    
    def get_temperature(self):
        """Get CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return 0.0
    
    def generate_report(self):
        """Generate benchmark report"""
        print("AI Performance Benchmark Report")
        print("=" * 50)
        
        for result in self.results:
            print(f"\nModel: {result.model_name}")
            print(f"Inference Time: {result.avg_inference_time:.3f}s")
            print(f"FPS: {result.fps:.1f}")
            print(f"CPU Usage: {result.cpu_usage:.1f}%")
            print(f"Memory Usage: {result.memory_usage:.1f}%")
            print(f"Temperature: {result.temperature:.1f}°C")

# Usage
benchmark = AIBenchmark()
result = benchmark.benchmark_model('model.tflite', (1, 224, 224, 3))
benchmark.generate_report()
```

### Real-World Performance Tests
```python
class VisionSystemBenchmark:
    def __init__(self, camera_resolution=(640, 480)):
        """Benchmark complete vision system"""
        self.camera_resolution = camera_resolution
        
    def benchmark_end_to_end(self, model_path, num_frames=100):
        """Benchmark complete pipeline including camera capture"""
        from picamera2 import Picamera2
        
        # Initialize camera and model
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"format": 'RGB888', "size": self.camera_resolution}
        )
        picam2.configure(config)
        
        model = TFLiteModel(model_path)
        
        # Benchmark complete pipeline
        picam2.start()
        
        total_time = 0
        for i in range(num_frames):
            start_time = time.time()
            
            # Capture frame
            frame = picam2.capture_array()
            
            # Preprocess
            processed = cv2.resize(frame, (224, 224))
            processed = processed / 255.0
            
            # Inference
            result = model.predict(processed)
            
            # Postprocess (example: get top prediction)
            prediction = np.argmax(result)
            
            frame_time = time.time() - start_time
            total_time += frame_time
        
        picam2.stop()
        
        avg_time = total_time / num_frames
        fps = 1.0 / avg_time
        
        print(f"End-to-end performance:")
        print(f"Average time per frame: {avg_time:.3f}s")
        print(f"Achievable FPS: {fps:.1f}")
        
        return avg_time, fps
```

## Optimization Guidelines

### Model Selection Criteria
1. **Accuracy vs Speed**: Balance for application requirements
2. **Model Size**: Consider memory constraints
3. **Input Resolution**: Lower resolution = faster inference
4. **Architecture**: Prefer mobile-optimized architectures

### Hardware Configuration
```bash
# Optimize Raspberry Pi for AI workloads

# 1. Increase GPU memory split
echo 'gpu_mem=128' >> /boot/config.txt

# 2. Enable performance governor
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 3. Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-country

# 4. Optimize memory usage
echo 'vm.swappiness=1' >> /etc/sysctl.conf
```

### Software Optimization
```python
# Performance tips for Python AI applications

# 1. Use NumPy operations instead of Python loops
import numpy as np

# Bad
result = []
for i in range(len(data)):
    result.append(data[i] * 2)

# Good
result = data * 2

# 2. Minimize memory allocations
def process_batch(images):
    """Process batch efficiently"""
    # Pre-allocate output array
    batch_size = len(images)
    output = np.zeros((batch_size, 224, 224, 3))
    
    for i, img in enumerate(images):
        output[i] = preprocess_image(img)
    
    return output

# 3. Use appropriate data types
input_data = np.array(image, dtype=np.float32)  # TFLite prefers float32
```

## Practical Exercises

### Exercise 1: Model Conversion and Optimization
1. Convert a Keras model to TensorFlow Lite
2. Apply post-training quantization
3. Compare performance and accuracy
4. Measure inference time and resource usage

### Exercise 2: Hardware Accelerator Integration
1. Set up external AI accelerator (if available)
2. Convert model to accelerator format
3. Benchmark performance improvement
4. Compare power consumption

### Exercise 3: Real-Time Vision Pipeline
1. Build complete camera → AI → output pipeline
2. Optimize for real-time performance
3. Implement frame dropping for consistent timing
4. Add performance monitoring

### Exercise 4: Edge AI Deployment
1. Create production-ready inference service
2. Add model versioning and hot-swapping
3. Implement health monitoring
4. Test thermal performance under load

## References

1. [TensorFlow Lite Optimization Guide](https://www.tensorflow.org/lite/performance)
2. [Google Coral Documentation](https://coral.ai/docs/)
3. [OpenVINO Toolkit](https://docs.openvino.ai/)
4. [Raspberry Pi AI Benchmarks](https://github.com/AI-benchmarks/RPi-AI-Benchmark)

---

**Next Section**: [Vision System →](../04-vision-system/README.md)