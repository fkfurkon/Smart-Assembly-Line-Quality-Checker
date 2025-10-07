# Vision System for Assembly Line

## Overview
This section covers the development of a complete computer vision system for assembly line applications using Raspberry Pi. We'll implement practical solutions for quality control, object detection, defect identification, and automated inspection processes.

## Table of Contents
1. [Industrial Vision Requirements](#industrial-vision-requirements)
2. [System Architecture](#system-architecture)
3. [Camera Setup and Calibration](#camera-setup-and-calibration)
4. [Image Processing Pipeline](#image-processing-pipeline)
5. [Object Detection and Classification](#object-detection-and-classification)
6. [Quality Control Applications](#quality-control-applications)
7. [Integration with Assembly Line](#integration-with-assembly-line)

## Industrial Vision Requirements

### Performance Requirements
- **Real-time Processing**: 10-30 FPS depending on application
- **Accuracy**: >95% detection rate, <1% false positive rate
- **Reliability**: 24/7 operation capability
- **Repeatability**: Consistent results across environmental conditions
- **Response Time**: <100ms for critical safety applications

### Environmental Considerations
- **Lighting Variations**: Fluorescent, LED, natural light changes
- **Vibration**: Mechanical isolation requirements
- **Temperature**: Operating range -10°C to +60°C
- **Dust and Contamination**: IP65 rated enclosures
- **Electromagnetic Interference**: Shielding from motors and welders

### Integration Requirements
- **PLC Communication**: Modbus, Ethernet/IP, or digital I/O
- **HMI Interface**: Web-based or touchscreen displays
- **Database Logging**: Production data and quality metrics
- **Alarm Systems**: Visual and audible notifications
- **Network Integration**: Integration with MES/ERP systems

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────┐
│                Assembly Line Vision System              │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Camera     │  │   Lighting  │  │   Triggers      │ │
│  │   Module     │  │   System    │  │   & Sensors     │ │
│  └──────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Raspberry Pi │  │ AI Models   │  │   GPIO I/O      │ │
│  │  Processing  │  │ TFLite/CV   │  │   Interface     │ │
│  │    Unit      │  │             │  │                 │ │
│  └──────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Data        │  │   Network   │  │   Control       │ │
│  │  Logging     │  │   Interface │  │   Outputs       │ │
│  │              │  │   (Ethernet)│  │   (Relays)      │ │
│  └──────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Software Architecture
```python
# vision_system/core/architecture.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import threading
import queue
import time

@dataclass
class VisionResult:
    """Standard result format for vision operations"""
    timestamp: float
    object_detected: bool
    classification: str
    confidence: float
    bounding_box: Optional[tuple]
    defects: list
    pass_fail: bool
    processing_time: float

class VisionModule(ABC):
    """Abstract base class for vision processing modules"""
    
    @abstractmethod
    def process(self, image) -> VisionResult:
        """Process image and return results"""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]):
        """Configure module parameters"""
        pass

class VisionSystem:
    """Main vision system coordinator"""
    
    def __init__(self):
        self.modules = {}
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=100)
        self.running = False
        self.worker_thread = None
    
    def add_module(self, name: str, module: VisionModule):
        """Add processing module to system"""
        self.modules[name] = module
    
    def start(self):
        """Start vision system processing"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_loop)
        self.worker_thread.start()
    
    def stop(self):
        """Stop vision system processing"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get image from queue with timeout
                image, metadata = self.input_queue.get(timeout=1.0)
                
                # Process through all modules
                results = {}
                for name, module in self.modules.items():
                    try:
                        result = module.process(image)
                        results[name] = result
                    except Exception as e:
                        print(f"Error in module {name}: {e}")
                
                # Combine results and send to output
                combined_result = self._combine_results(results, metadata)
                self.output_queue.put(combined_result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def _combine_results(self, results: Dict, metadata: Dict) -> VisionResult:
        """Combine results from multiple modules"""
        # Implementation depends on specific application logic
        pass
```

## Camera Setup and Calibration

### Camera Selection and Mounting
```python
# vision_system/hardware/camera.py
from picamera2 import Picamera2
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    resolution: tuple = (1920, 1080)
    framerate: int = 30
    exposure_mode: str = 'auto'
    awb_mode: str = 'auto'
    brightness: int = 50
    contrast: int = 0
    saturation: int = 0
    sharpness: int = 0

class IndustrialCamera:
    """Industrial camera interface for Raspberry Pi"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera = None
        self.calibration_data = None
        
    def initialize(self):
        """Initialize camera with industrial settings"""
        self.camera = Picamera2()
        
        # Configure for consistent industrial imaging
        camera_config = self.camera.create_video_configuration(
            main={"format": 'RGB888', "size": self.config.resolution},
            controls={
                "Brightness": self.config.brightness / 100.0,
                "Contrast": self.config.contrast / 100.0 + 1.0,
                "Saturation": self.config.saturation / 100.0 + 1.0,
                "Sharpness": self.config.sharpness / 100.0 + 1.0,
                # Fixed settings for repeatability
                "ExposureTime": 10000,  # 10ms exposure
                "AnalogueGain": 1.0,
                "AwbEnable": False,  # Disable auto white balance
                "ColourGains": (1.4, 1.5),  # Fixed color gains
            }
        )
        
        self.camera.configure(camera_config)
        self.camera.start()
        
        # Allow camera to settle
        time.sleep(2)
    
    def capture_frame(self):
        """Capture single frame with error handling"""
        try:
            frame = self.camera.capture_array()
            return frame
        except Exception as e:
            print(f"Camera capture error: {e}")
            return None
    
    def calibrate_camera(self, calibration_images):
        """Perform camera calibration for metric measurements"""
        # Chessboard calibration
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Chessboard dimensions
        chessboard_size = (9, 6)
        
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points
        imgpoints = []  # 2D points
        
        for img_path in calibration_images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        self.calibration_data = {
            'camera_matrix': mtx,
            'distortion_coefficients': dist,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs
        }
        
        return self.calibration_data
    
    def undistort_image(self, image):
        """Remove lens distortion from image"""
        if self.calibration_data is None:
            return image
        
        mtx = self.calibration_data['camera_matrix']
        dist = self.calibration_data['distortion_coefficients']
        
        undistorted = cv2.undistort(image, mtx, dist, None, mtx)
        return undistorted
```

### Lighting System Integration
```python
# vision_system/hardware/lighting.py
from gpiozero import PWMOutputDevice
import time

class IndustrialLighting:
    """Control industrial LED lighting system"""
    
    def __init__(self, led_pins: list):
        self.led_controllers = []
        for pin in led_pins:
            self.led_controllers.append(PWMOutputDevice(pin))
        
        self.current_intensity = 0.8  # 80% default intensity
        
    def set_intensity(self, intensity: float):
        """Set LED intensity (0.0 to 1.0)"""
        self.current_intensity = max(0.0, min(1.0, intensity))
        for led in self.led_controllers:
            led.value = self.current_intensity
    
    def strobe_lighting(self, duration_ms: int = 50):
        """Provide strobe lighting for motion freeze"""
        # Turn off
        for led in self.led_controllers:
            led.value = 0
        
        time.sleep(0.01)  # Brief dark period
        
        # Full intensity strobe
        for led in self.led_controllers:
            led.value = 1.0
        
        time.sleep(duration_ms / 1000.0)
        
        # Return to normal intensity
        for led in self.led_controllers:
            led.value = self.current_intensity
    
    def automatic_brightness_control(self, target_brightness: int = 128):
        """Adjust lighting based on image brightness feedback"""
        # This would integrate with camera feedback
        # to maintain consistent illumination
        pass
```

## Image Processing Pipeline

### Basic Image Processing Operations
```python
# vision_system/processing/image_ops.py
import cv2
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    """Core image processing operations for industrial vision"""
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
        """Enhance image contrast and brightness"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def reduce_noise(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Apply noise reduction filters"""
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            return image
    
    @staticmethod
    def edge_detection(image: np.ndarray, method: str = 'canny') -> np.ndarray:
        """Detect edges in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == 'canny':
            return cv2.Canny(gray, 50, 150, apertureSize=3)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        elif method == 'laplacian':
            return cv2.Laplacian(gray, cv2.CV_64F)
    
    @staticmethod
    def threshold_image(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """Apply thresholding for binary segmentation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == 'adaptive':
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == 'fixed':
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return binary
    
    @staticmethod
    def morphological_operations(image: np.ndarray, operation: str = 'opening', 
                               kernel_size: int = 5) -> np.ndarray:
        """Apply morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'erosion':
            return cv2.erode(image, kernel, iterations=1)
        elif operation == 'dilation':
            return cv2.dilate(image, kernel, iterations=1)
        
        return image

class ROIManager:
    """Manage Regions of Interest for focused processing"""
    
    def __init__(self):
        self.roi_definitions = {}
    
    def define_roi(self, name: str, x: int, y: int, width: int, height: int):
        """Define a region of interest"""
        self.roi_definitions[name] = (x, y, width, height)
    
    def extract_roi(self, image: np.ndarray, roi_name: str) -> Optional[np.ndarray]:
        """Extract ROI from image"""
        if roi_name not in self.roi_definitions:
            return None
        
        x, y, w, h = self.roi_definitions[roi_name]
        return image[y:y+h, x:x+w]
    
    def overlay_roi(self, image: np.ndarray, roi_name: str, 
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw ROI rectangle on image"""
        if roi_name not in self.roi_definitions:
            return image
        
        x, y, w, h = self.roi_definitions[roi_name]
        result = image.copy()
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, roi_name, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result
```

### Advanced Processing Techniques
```python
# vision_system/processing/advanced_ops.py
import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

class AdvancedProcessor:
    """Advanced image processing for industrial applications"""
    
    @staticmethod
    def template_matching(image: np.ndarray, template: np.ndarray, 
                         threshold: float = 0.8) -> list:
        """Find template matches in image"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        
        # Perform template matching
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # Find matches above threshold
        locations = np.where(result >= threshold)
        matches = []
        
        template_h, template_w = gray_template.shape
        
        for pt in zip(*locations[::-1]):
            matches.append({
                'position': pt,
                'confidence': result[pt[1], pt[0]],
                'bounding_box': (pt[0], pt[1], template_w, template_h)
            })
        
        return matches
    
    @staticmethod
    def blob_detection(image: np.ndarray, min_area: int = 100, 
                      max_area: int = 10000) -> list:
        """Detect and analyze blobs in binary image"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                # Calculate blob properties
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = 0, 0
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional properties
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                blobs.append({
                    'centroid': (cx, cy),
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'bounding_box': (x, y, w, h),
                    'contour': contour
                })
        
        return blobs
    
    @staticmethod
    def color_segmentation(image: np.ndarray, num_clusters: int = 3) -> np.ndarray:
        """Segment image by color using K-means clustering"""
        # Reshape image to feature vector
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        
        # Convert back to image
        segmented = labels.reshape(image.shape[:2])
        
        return segmented.astype(np.uint8)
    
    @staticmethod
    def measure_dimensions(contour: np.ndarray, pixels_per_mm: float) -> dict:
        """Measure physical dimensions from contour"""
        # Minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate dimensions
        width_pixels = rect[1][0]
        height_pixels = rect[1][1]
        
        # Convert to physical units
        width_mm = width_pixels / pixels_per_mm
        height_mm = height_pixels / pixels_per_mm
        area_mm2 = cv2.contourArea(contour) / (pixels_per_mm ** 2)
        
        return {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'area_mm2': area_mm2,
            'angle_degrees': rect[2],
            'bounding_box': box
        }
```

## Object Detection and Classification

### Traditional Computer Vision Approach
```python
# vision_system/detection/traditional_cv.py
import cv2
import numpy as np
from typing import List, Dict

class TraditionalDetector:
    """Traditional computer vision object detection"""
    
    def __init__(self):
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.reference_features = {}
    
    def train_reference_object(self, name: str, reference_image: np.ndarray):
        """Train detector with reference object"""
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        self.reference_features[name] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'image': reference_image
        }
    
    def detect_objects(self, image: np.ndarray, min_matches: int = 10) -> List[Dict]:
        """Detect trained objects in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        detections = []
        
        for object_name, ref_data in self.reference_features.items():
            if descriptors is None or ref_data['descriptors'] is None:
                continue
            
            # Match features
            matches = self.matcher.knnMatch(ref_data['descriptors'], descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) >= min_matches:
                # Extract matched keypoints
                src_pts = np.float32([ref_data['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if homography is not None:
                    # Get object corners
                    h, w = ref_data['image'].shape[:2]
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, homography)
                    
                    detections.append({
                        'name': object_name,
                        'corners': transformed_corners,
                        'matches': len(good_matches),
                        'confidence': len(good_matches) / len(ref_data['keypoints'])
                    })
        
        return detections

class ShapeDetector:
    """Detect geometric shapes"""
    
    @staticmethod
    def detect_circles(image: np.ndarray, min_radius: int = 20, max_radius: int = 100) -> List[Dict]:
        """Detect circular objects"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_radius * 2,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detected_circles.append({
                    'center': (x, y),
                    'radius': r,
                    'area': np.pi * r * r,
                    'type': 'circle'
                })
        
        return detected_circles
    
    @staticmethod
    def detect_rectangles(image: np.ndarray, min_area: int = 1000) -> List[Dict]:
        """Detect rectangular objects"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection and contour finding
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 corners)
            if len(approx) == 4:
                # Calculate aspect ratio and other properties
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                rectangles.append({
                    'corners': approx,
                    'bounding_box': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'type': 'rectangle'
                })
        
        return rectangles
```

### AI-Based Object Detection
```python
# vision_system/detection/ai_detector.py
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from typing import List, Dict, Tuple

class AIObjectDetector:
    """AI-based object detection using TensorFlow Lite"""
    
    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load class labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize image
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in image"""
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get outputs
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Process detections
        detections = []
        image_height, image_width = image.shape[:2]
        
        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                x1 = int(xmin * image_width)
                y1 = int(ymin * image_height)
                x2 = int(xmax * image_width)
                y2 = int(ymax * image_height)
                
                class_id = int(classes[i])
                class_name = self.labels[class_id] if class_id < len(self.labels) else 'Unknown'
                
                detections.append({
                    'class_name': class_name,
                    'class_id': class_id,
                    'confidence': float(scores[i]),
                    'bounding_box': (x1, y1, x2 - x1, y2 - y1),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        return detections

class DefectDetector:
    """Specialized defect detection for quality control"""
    
    def __init__(self, model_path: str):
        self.model = AIObjectDetector(model_path, 'defect_labels.txt', confidence_threshold=0.3)
        
        # Define defect types and severity levels
        self.defect_severity = {
            'scratch': 'minor',
            'dent': 'major',
            'crack': 'critical',
            'discoloration': 'minor',
            'missing_part': 'critical'
        }
    
    def inspect_part(self, image: np.ndarray) -> Dict:
        """Comprehensive part inspection"""
        detections = self.model.detect_objects(image)
        
        # Categorize defects
        defects = []
        critical_count = 0
        major_count = 0
        minor_count = 0
        
        for detection in detections:
            defect_type = detection['class_name']
            severity = self.defect_severity.get(defect_type, 'unknown')
            
            defect_info = {
                'type': defect_type,
                'severity': severity,
                'confidence': detection['confidence'],
                'location': detection['bounding_box'],
                'area': detection['bounding_box'][2] * detection['bounding_box'][3]
            }
            
            defects.append(defect_info)
            
            # Count by severity
            if severity == 'critical':
                critical_count += 1
            elif severity == 'major':
                major_count += 1
            elif severity == 'minor':
                minor_count += 1
        
        # Determine pass/fail status
        pass_fail = (critical_count == 0 and major_count <= 1 and minor_count <= 3)
        
        return {
            'defects': defects,
            'defect_counts': {
                'critical': critical_count,
                'major': major_count,
                'minor': minor_count
            },
            'pass_fail': pass_fail,
            'overall_confidence': np.mean([d['confidence'] for d in defects]) if defects else 1.0
        }
```

## Quality Control Applications

### Dimensional Inspection
```python
# vision_system/quality/dimensional.py
import cv2
import numpy as np
from typing import Dict, List, Tuple

class DimensionalInspector:
    """Precision dimensional measurement and inspection"""
    
    def __init__(self, pixels_per_mm: float, tolerance_mm: float = 0.1):
        self.pixels_per_mm = pixels_per_mm
        self.tolerance_mm = tolerance_mm
        
    def calibrate_pixel_scale(self, calibration_image: np.ndarray, 
                            known_distance_mm: float) -> float:
        """Calibrate pixel-to-millimeter conversion"""
        # Detect calibration object (e.g., reference ruler or known part)
        gray = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours and identify calibration object
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be calibration object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            _, _, width_pixels, _ = cv2.boundingRect(largest_contour)
            
            # Calculate pixels per mm
            self.pixels_per_mm = width_pixels / known_distance_mm
            
        return self.pixels_per_mm
    
    def measure_part_dimensions(self, image: np.ndarray, part_contour: np.ndarray) -> Dict:
        """Measure part dimensions with high precision"""
        # Minimum area rectangle for overall dimensions
        rect = cv2.minAreaRect(part_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate dimensions
        width_pixels = rect[1][0]
        height_pixels = rect[1][1]
        
        # Convert to physical units
        width_mm = width_pixels / self.pixels_per_mm
        height_mm = height_pixels / self.pixels_per_mm
        area_mm2 = cv2.contourArea(part_contour) / (self.pixels_per_mm ** 2)
        
        # Calculate additional measurements
        perimeter_pixels = cv2.arcLength(part_contour, True)
        perimeter_mm = perimeter_pixels / self.pixels_per_mm
        
        # Circularity calculation
        circularity = 4 * np.pi * area_mm2 / (perimeter_mm ** 2) if perimeter_mm > 0 else 0
        
        return {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'area_mm2': area_mm2,
            'perimeter_mm': perimeter_mm,
            'circularity': circularity,
            'angle_degrees': rect[2],
            'bounding_box': box
        }
    
    def check_tolerances(self, measured_dims: Dict, target_dims: Dict) -> Dict:
        """Check if dimensions are within tolerance"""
        results = {}
        
        for dim_name, target_value in target_dims.items():
            if dim_name in measured_dims:
                measured_value = measured_dims[dim_name]
                deviation = abs(measured_value - target_value)
                within_tolerance = deviation <= self.tolerance_mm
                
                results[dim_name] = {
                    'measured': measured_value,
                    'target': target_value,
                    'deviation': deviation,
                    'within_tolerance': within_tolerance,
                    'tolerance_limit': self.tolerance_mm
                }
        
        # Overall pass/fail
        overall_pass = all(result['within_tolerance'] for result in results.values())
        results['overall_pass'] = overall_pass
        
        return results

class SurfaceInspector:
    """Surface quality and finish inspection"""
    
    def __init__(self):
        self.surface_metrics = {}
    
    def analyze_surface_roughness(self, roi_image: np.ndarray) -> Dict:
        """Analyze surface roughness using texture analysis"""
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image
        
        # Calculate texture metrics
        # Standard deviation (roughness indicator)
        roughness = np.std(gray)
        
        # Local binary pattern for texture analysis
        lbp = self._local_binary_pattern(gray)
        texture_uniformity = self._calculate_uniformity(lbp)
        
        # Edge density (another roughness measure)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'roughness_std': roughness,
            'texture_uniformity': texture_uniformity,
            'edge_density': edge_density,
            'surface_quality': self._classify_surface_quality(roughness, edge_density)
        }
    
    def _local_binary_pattern(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                pattern = 0
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(radius * np.cos(angle))
                    y = int(radius * np.sin(angle))
                    
                    neighbor = image[i + y, j + x]
                    if neighbor >= center:
                        pattern |= (1 << k)
                
                lbp[i, j] = pattern
        
        return lbp
    
    def _calculate_uniformity(self, lbp: np.ndarray) -> float:
        """Calculate texture uniformity from LBP"""
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= hist.sum()
        
        # Shannon entropy (lower = more uniform)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        uniformity = 1 / (1 + entropy)  # Convert to uniformity measure
        
        return uniformity
    
    def _classify_surface_quality(self, roughness: float, edge_density: float) -> str:
        """Classify surface quality based on metrics"""
        if roughness < 10 and edge_density < 0.1:
            return 'excellent'
        elif roughness < 20 and edge_density < 0.2:
            return 'good'
        elif roughness < 35 and edge_density < 0.35:
            return 'acceptable'
        else:
            return 'poor'
```

## Integration with Assembly Line

### PLC Communication
```python
# vision_system/integration/plc_interface.py
import socket
import struct
import time
from typing import Dict, Any
import threading

class ModbusTCPClient:
    """Modbus TCP communication with PLC"""
    
    def __init__(self, plc_ip: str, plc_port: int = 502, unit_id: int = 1):
        self.plc_ip = plc_ip
        self.plc_port = plc_port
        self.unit_id = unit_id
        self.socket = None
        self.transaction_id = 0
        
    def connect(self) -> bool:
        """Connect to PLC"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.plc_ip, self.plc_port))
            return True
        except Exception as e:
            print(f"PLC connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from PLC"""
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def read_holding_registers(self, start_address: int, count: int) -> list:
        """Read holding registers from PLC"""
        if not self.socket:
            return []
        
        try:
            # Build Modbus TCP frame
            self.transaction_id += 1
            
            # MBAP Header
            mbap = struct.pack('>HHHB', 
                              self.transaction_id,  # Transaction ID
                              0,                    # Protocol ID
                              6,                    # Length
                              self.unit_id)         # Unit ID
            
            # PDU (Protocol Data Unit)
            pdu = struct.pack('>BHH', 
                             3,              # Function code (Read Holding Registers)
                             start_address,  # Starting address
                             count)          # Quantity
            
            # Send request
            request = mbap + pdu
            self.socket.send(request)
            
            # Receive response
            response = self.socket.recv(1024)
            
            # Parse response
            if len(response) >= 9:
                byte_count = response[8]
                data_start = 9
                data_end = data_start + byte_count
                
                values = []
                for i in range(data_start, data_end, 2):
                    value = struct.unpack('>H', response[i:i+2])[0]
                    values.append(value)
                
                return values
            
        except Exception as e:
            print(f"Modbus read error: {e}")
        
        return []
    
    def write_single_register(self, address: int, value: int) -> bool:
        """Write single register to PLC"""
        if not self.socket:
            return False
        
        try:
            # Build Modbus TCP frame
            self.transaction_id += 1
            
            # MBAP Header
            mbap = struct.pack('>HHHB', 
                              self.transaction_id,
                              0,
                              6,
                              self.unit_id)
            
            # PDU
            pdu = struct.pack('>BHH', 
                             6,        # Function code (Write Single Register)
                             address,  # Register address
                             value)    # Register value
            
            # Send request
            request = mbap + pdu
            self.socket.send(request)
            
            # Receive response
            response = self.socket.recv(1024)
            return len(response) >= 12
            
        except Exception as e:
            print(f"Modbus write error: {e}")
            return False

class AssemblyLineInterface:
    """Interface between vision system and assembly line"""
    
    def __init__(self, plc_ip: str):
        self.plc = ModbusTCPClient(plc_ip)
        self.running = False
        self.communication_thread = None
        
        # Define register mappings
        self.register_map = {
            'part_present': 100,      # Input: Part detection sensor
            'trigger_inspection': 101, # Input: Trigger from PLC
            'inspection_result': 200,  # Output: Pass(1)/Fail(0)
            'defect_count': 201,      # Output: Number of defects
            'system_ready': 202,      # Output: Vision system status
            'alarm_status': 203       # Output: Alarm condition
        }
        
        self.system_status = {
            'ready': True,
            'last_result': None,
            'inspection_count': 0,
            'error_count': 0
        }
    
    def start_communication(self):
        """Start PLC communication thread"""
        if self.plc.connect():
            self.running = True
            self.communication_thread = threading.Thread(target=self._communication_loop)
            self.communication_thread.start()
            
            # Signal system ready
            self.plc.write_single_register(self.register_map['system_ready'], 1)
        else:
            print("Failed to connect to PLC")
    
    def stop_communication(self):
        """Stop PLC communication"""
        self.running = False
        if self.communication_thread:
            self.communication_thread.join()
        self.plc.disconnect()
    
    def _communication_loop(self):
        """Main communication loop"""
        while self.running:
            try:
                # Read input registers
                inputs = self.plc.read_holding_registers(100, 10)
                
                if inputs:
                    part_present = inputs[0]
                    trigger_inspection = inputs[1]
                    
                    # Process inspection trigger
                    if trigger_inspection and part_present:
                        self._handle_inspection_trigger()
                
                # Update system status
                self._update_system_status()
                
                time.sleep(0.1)  # 100ms cycle time
                
            except Exception as e:
                print(f"Communication loop error: {e}")
                self.system_status['error_count'] += 1
    
    def _handle_inspection_trigger(self):
        """Handle inspection trigger from PLC"""
        # This would trigger the vision system inspection
        # Results would be sent back to PLC
        pass
    
    def _update_system_status(self):
        """Update system status registers"""
        try:
            # Update system ready status
            ready_status = 1 if self.system_status['ready'] else 0
            self.plc.write_single_register(self.register_map['system_ready'], ready_status)
            
            # Update alarm status
            alarm_status = 1 if self.system_status['error_count'] > 10 else 0
            self.plc.write_single_register(self.register_map['alarm_status'], alarm_status)
            
        except Exception as e:
            print(f"Status update error: {e}")
    
    def send_inspection_result(self, result: Dict):
        """Send inspection results to PLC"""
        try:
            # Send pass/fail result
            pass_fail = 1 if result.get('pass_fail', False) else 0
            self.plc.write_single_register(self.register_map['inspection_result'], pass_fail)
            
            # Send defect count
            defect_count = result.get('defect_count', 0)
            self.plc.write_single_register(self.register_map['defect_count'], defect_count)
            
            # Update statistics
            self.system_status['last_result'] = result
            self.system_status['inspection_count'] += 1
            
        except Exception as e:
            print(f"Result transmission error: {e}")
```

### Data Logging and Reporting
```python
# vision_system/integration/data_logger.py
import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, List
import threading

class ProductionDataLogger:
    """Log production data and quality metrics"""
    
    def __init__(self, database_path: str = 'production_data.db'):
        self.database_path = database_path
        self.connection = None
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                part_id TEXT,
                result TEXT,
                pass_fail INTEGER,
                defect_count INTEGER,
                processing_time REAL,
                confidence REAL,
                defects_json TEXT,
                image_path TEXT
            )
        ''')
        
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS production_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                shift_id TEXT,
                parts_inspected INTEGER,
                parts_passed INTEGER,
                parts_failed INTEGER,
                average_processing_time REAL,
                system_uptime REAL
            )
        ''')
        
        self.connection.commit()
    
    def log_inspection(self, result: Dict):
        """Log individual inspection result"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO inspections 
                (timestamp, part_id, result, pass_fail, defect_count, 
                 processing_time, confidence, defects_json, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                result.get('part_id', ''),
                result.get('classification', ''),
                1 if result.get('pass_fail', False) else 0,
                result.get('defect_count', 0),
                result.get('processing_time', 0.0),
                result.get('confidence', 0.0),
                json.dumps(result.get('defects', [])),
                result.get('image_path', '')
            ))
            self.connection.commit()
            
        except Exception as e:
            print(f"Database logging error: {e}")
    
    def get_production_summary(self, hours: int = 24) -> Dict:
        """Get production summary for specified hours"""
        try:
            cursor = self.connection.cursor()
            start_time = time.time() - (hours * 3600)
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_inspections,
                    SUM(pass_fail) as passed_parts,
                    COUNT(*) - SUM(pass_fail) as failed_parts,
                    AVG(processing_time) as avg_processing_time,
                    AVG(confidence) as avg_confidence
                FROM inspections 
                WHERE timestamp >= ?
            ''', (start_time,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'total_inspections': result[0],
                    'passed_parts': result[1] or 0,
                    'failed_parts': result[2] or 0,
                    'pass_rate': (result[1] or 0) / result[0] if result[0] > 0 else 0,
                    'avg_processing_time': result[3] or 0,
                    'avg_confidence': result[4] or 0
                }
            
        except Exception as e:
            print(f"Database query error: {e}")
        
        return {}
    
    def export_production_report(self, start_time: float, end_time: float, 
                               output_file: str):
        """Export production report to CSV"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT timestamp, part_id, result, pass_fail, defect_count,
                       processing_time, confidence
                FROM inspections 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start_time, end_time))
            
            results = cursor.fetchall()
            
            with open(output_file, 'w') as f:
                # Write header
                f.write('Timestamp,Part ID,Result,Pass/Fail,Defect Count,Processing Time,Confidence\n')
                
                # Write data
                for row in results:
                    timestamp_str = datetime.fromtimestamp(row[0]).isoformat()
                    f.write(f'{timestamp_str},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]:.3f},{row[6]:.3f}\n')
                    
        except Exception as e:
            print(f"Export error: {e}")
```

## Practical Exercises

### Exercise 1: Basic Vision System Setup
1. Set up camera with proper mounting and lighting
2. Implement basic image capture and display
3. Add ROI definition and processing
4. Test with sample objects

### Exercise 2: Object Detection Implementation
1. Train traditional CV detector with reference objects
2. Implement AI-based detection using pre-trained model
3. Compare performance and accuracy
4. Optimize for real-time operation

### Exercise 3: Quality Control System
1. Create dimensional inspection module
2. Implement defect detection algorithm
3. Add pass/fail decision logic
4. Test with various part samples

### Exercise 4: Assembly Line Integration
1. Set up PLC communication interface
2. Implement trigger-based inspection workflow
3. Add data logging and reporting
4. Create operator interface

## Performance Metrics

### Key Performance Indicators (KPIs)
- **Throughput**: Parts inspected per hour
- **Accuracy**: Detection rate and false positive rate
- **Availability**: System uptime percentage
- **Response Time**: Time from trigger to result
- **Quality**: Defect detection effectiveness

### Benchmark Targets
- Processing Time: <200ms per part
- Detection Accuracy: >98%
- False Positive Rate: <2%
- System Uptime: >95%
- Throughput: 1000+ parts/hour

## References

1. [Industrial Computer Vision Handbook](https://www.visiononline.org/)
2. [OpenCV Industrial Applications](https://opencv.org/applications/)
3. [Assembly Line Automation Guidelines](https://www.isa.org/)
4. [Machine Vision Lighting Guide](https://www.keyence.com/ss/products/vision/lighting/)

---

**Next Section**: [Electrical Wiring →](../05-electrical-wiring/README.md)