#!/usr/bin/env python3
"""
YOLO Bottle Detection System
============================
Advanced bottle detection using YOLO (You Only Look Once) deep learning model.
Supports YOLOv5, YOLOv8, and custom trained models for bottle detection.

Features:
- Real-time object detection with YOLO
- Pre-trained models for bottle/cup detection
- Custom bottle classification
- High accuracy and speed
- Confidence scoring and filtering
- Multi-class detection support

Requirements:
- ultralytics (for YOLOv8)
- torch
- opencv-python
- numpy

Controls:
- SPACE: Detect bottles in current frame
- S: Save frame with detections
- C: Toggle continuous detection
- M: Switch between YOLO models
- T: Adjust confidence threshold
- R: Reset statistics
- Q: Quit

Author: Cooperative Education Project
Date: September 2025
"""

import cv2
import numpy as np
import time
import json
import logging
from datetime import datetime
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")

class YOLOBottleDetector:
    def __init__(self, camera_id=0):
        """Initialize YOLO bottle detector"""
        self.camera_id = camera_id
        self.cap = None
        self.model = None
        self.model_name = "yolov8n.pt"  # Default model
        self.available_models = [
            "yolov8n.pt",    # Nano - fastest
            "yolov8s.pt",    # Small
            "yolov8m.pt",    # Medium
            "yolov8l.pt",    # Large - most accurate
        ]
        self.current_model_index = 0
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 100
        
        # Target classes for bottle detection
        self.bottle_classes = {
            39: 'bottle',      # COCO dataset bottle class
            41: 'cup',         # COCO dataset cup class  
            # Add more classes as needed
        }
        
        # Detection statistics
        self.detection_stats = {
            'total_detections': 0,
            'bottles_detected': 0,
            'session_start': datetime.now().isoformat(),
            'model_used': self.model_name
        }
        
        self.continuous_mode = False
        self.last_detection_time = 0
        self.center_tolerance = 100
        
        # For smooth continuous display
        self.last_display_frame = None
        self.detection_interval = 0.3  # seconds between detections in continuous mode
        self.display_update_rate = 30  # FPS for smooth display
        
        # Initialize camera and model
        self.init_camera()
        if YOLO_AVAILABLE:
            self.init_yolo_model()
        else:
            logger.error("YOLO not available. Please install ultralytics package.")
            
    def init_camera(self):
        """Initialize camera connection"""
        logger.info(f"Initializing camera {self.camera_id}...")
        
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info(f"Camera initialized successfully")
                        logger.info(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                        return
                        
            except Exception as e:
                logger.warning(f"Failed to initialize camera with backend {backend}: {e}")
                continue
                
        raise RuntimeError("Camera initialization failed")
        
    def init_yolo_model(self):
        """Initialize YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            logger.info(f"YOLO model loaded successfully")
            logger.info(f"Model classes: {len(self.model.names)} classes available")
            
            # Print available classes related to bottles
            bottle_related_classes = []
            for class_id, class_name in self.model.names.items():
                if any(keyword in class_name.lower() for keyword in ['bottle', 'cup', 'glass', 'drink']):
                    bottle_related_classes.append(f"{class_id}: {class_name}")
            
            if bottle_related_classes:
                logger.info(f"Bottle-related classes found: {bottle_related_classes}")
            else:
                logger.info("Using default bottle classes: 39 (bottle), 41 (cup)")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
            
    def switch_model(self):
        """Switch to next available YOLO model"""
        if not YOLO_AVAILABLE:
            return
            
        self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
        self.model_name = self.available_models[self.current_model_index]
        
        logger.info(f"Switching to model: {self.model_name}")
        self.init_yolo_model()
        self.detection_stats['model_used'] = self.model_name
        
    def detect_bottles_yolo(self, frame):
        """Detect bottles using YOLO model"""
        start_time = time.time()
        
        if self.model is None:
            return [], frame, 0
        
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            detected_bottles = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection data
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Check if detected class is bottle-related
                        class_name = self.model.names[class_id]
                        
                        # Filter for bottle-related objects
                        if (class_id in self.bottle_classes or 
                            any(keyword in class_name.lower() for keyword in ['bottle', 'cup'])):
                            
                            # Calculate center and dimensions
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            width = int(x2 - x1)
                            height = int(y2 - y1)
                            area = width * height
                            
                            # Calculate aspect ratio
                            aspect_ratio = height / width if width > 0 else 0
                            
                            detected_bottles.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), width, height),
                                'center': (center_x, center_y),
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'coordinates': (x1, y1, x2, y2)
                            })
            
            # Sort by confidence (highest first)
            detected_bottles.sort(key=lambda x: x['confidence'], reverse=True)
            
            processing_time = (time.time() - start_time) * 1000
            return detected_bottles, frame, processing_time
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return [], frame, 0
    
    def analyze_yolo_detections(self, bottles, frame_shape):
        """Analyze YOLO detection results"""
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        
        if len(bottles) == 0:
            return "FAIL", "ไม่พบขวดด้วย YOLO - No bottles detected by YOLO"
        
        # Take the highest confidence detection
        best_bottle = bottles[0]
        confidence = best_bottle['confidence']
        center_x, center_y = best_bottle['center']
        class_name = best_bottle['class_name']
        area = best_bottle['area']
        aspect_ratio = best_bottle['aspect_ratio']
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return "FAIL", f"ความน่าเชื่อถือต่ำ - Low confidence ({confidence:.2f})"
        
        # Check if bottle is centered
        distance_from_center = np.sqrt(
            (center_x - frame_center_x)**2 + (center_y - frame_center_y)**2
        )
        
        if distance_from_center > self.center_tolerance:
            return "FAIL", f"ขวดไม่อยู่กึ่งกลาง - Off-center ({distance_from_center:.1f}px)"
        
        # Check if multiple high-confidence bottles
        if len(bottles) > 1 and bottles[1]['confidence'] > 0.6:
            return "FAIL", f"พบขวดหลายใบ - Multiple bottles ({len(bottles)} detected)"
        
        # Successful detection
        return "PASS", f"YOLO ตรวจพบ {class_name} (conf: {confidence:.2f}, area: {area}px)"
    
    def draw_yolo_overlay(self, frame, bottles, status, reason, processing_time):
        """Draw YOLO detection overlay"""
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center crosshairs
        cv2.line(overlay_frame, (w//2-40, h//2), (w//2+40, h//2), (0, 255, 255), 3)
        cv2.line(overlay_frame, (w//2, h//2-40), (w//2, h//2+40), (0, 255, 255), 3)
        cv2.circle(overlay_frame, (w//2, h//2), self.center_tolerance, (0, 255, 255), 2)
        
        # Draw YOLO detections
        for i, bottle in enumerate(bottles):
            x1, y1, x2, y2 = bottle['coordinates']
            confidence = bottle['confidence']
            class_name = bottle['class_name']
            center = bottle['center']
            
            # Color based on confidence
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw bounding box
            cv2.rectangle(overlay_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw center point
            cv2.circle(overlay_frame, center, 6, color, -1)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay_frame, 
                         (int(x1), int(y1-25)), 
                         (int(x1 + label_size[0] + 10), int(y1)), 
                         color, -1)
            
            # Draw label text
            cv2.putText(overlay_frame, label, 
                       (int(x1+5), int(y1-5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw additional info
            info_text = f"Area: {bottle['area']}px | Ratio: {bottle['aspect_ratio']:.2f}"
            cv2.putText(overlay_frame, info_text, 
                       (int(x1), int(y2+20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status overlay
        status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        
        # Background for status
        cv2.rectangle(overlay_frame, (5, 5), (w-5, 130), (0, 0, 0), -1)
        
        cv2.putText(overlay_frame, f"YOLO BOTTLE DETECTION: {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(overlay_frame, f"Result: {reason}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Model info
        cv2.putText(overlay_frame, f"Model: {self.model_name} | Conf: {self.confidence_threshold}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(overlay_frame, f"Processing: {processing_time:.1f}ms | Bottles: {len(bottles)}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Stats
        mode_text = "CONTINUOUS" if self.continuous_mode else "MANUAL"
        cv2.putText(overlay_frame, f"Mode: {mode_text} | Total: {self.detection_stats['total_detections']}", 
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Controls
        cv2.putText(overlay_frame, "SPACE:Detect C:Continuous F:Speed M:Model T:Threshold S:Save R:Reset Q:Quit", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return overlay_frame
    
    def save_yolo_detection_result(self, bottles, status, reason, processing_time):
        """Save YOLO detection results"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'detection_type': 'yolo_bottle_detection',
            'model_used': self.model_name,
            'status': status,
            'reason': reason,
            'bottles_count': len(bottles),
            'processing_time_ms': processing_time,
            'confidence_threshold': self.confidence_threshold,
            'bottles': []
        }
        
        for bottle in bottles:
            result['bottles'].append({
                'class_id': bottle['class_id'],
                'class_name': bottle['class_name'],
                'confidence': float(bottle['confidence']),
                'center': bottle['center'],
                'bbox': bottle['bbox'],
                'area': bottle['area'],
                'aspect_ratio': float(bottle['aspect_ratio']),
                'coordinates': [float(x) for x in bottle['coordinates']]
            })
        
        filename = f"yolo_bottle_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"YOLO detection result saved to {filename}")
        return filename
    
    def adjust_confidence_threshold(self, increase=True):
        """Adjust confidence threshold"""
        if increase:
            self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
        else:
            self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
        
        logger.info(f"Confidence threshold adjusted to: {self.confidence_threshold:.2f}")
    
    def run(self):
        """Main YOLO detection loop"""
        if not YOLO_AVAILABLE:
            print("YOLO is not available. Please install ultralytics:")
            print("pip install ultralytics")
            return
            
        logger.info("Starting YOLO bottle detection...")
        logger.info(f"Using model: {self.model_name}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info("Controls: SPACE=Detect, C=Continuous, F=Speed, M=Model, T=Threshold, S=Save, R=Reset, Q=Quit")
        
        # Initialize display variables
        current_bottles = []
        current_status = ""
        current_reason = ""
        current_processing_time = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                should_detect = False
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    should_detect = True
                elif key == ord('c'):
                    self.continuous_mode = not self.continuous_mode
                    logger.info(f"Continuous YOLO detection: {'ON' if self.continuous_mode else 'OFF'}")
                    # Reset detection time when toggling continuous mode
                    self.last_detection_time = 0
                elif key == ord('m'):
                    self.switch_model()
                    # Clear previous detection results when switching models
                    current_bottles = []
                    current_status = ""
                    current_reason = ""
                elif key == ord('t'):
                    # Toggle between high and low confidence
                    if self.confidence_threshold >= 0.7:
                        self.confidence_threshold = 0.3
                    else:
                        self.confidence_threshold = 0.7
                    logger.info(f"Confidence threshold set to: {self.confidence_threshold}")
                elif key == ord('r'):
                    self.detection_stats = {
                        'total_detections': 0,
                        'bottles_detected': 0,
                        'session_start': datetime.now().isoformat(),
                        'model_used': self.model_name
                    }
                    # Clear display when resetting
                    current_bottles = []
                    current_status = ""
                    current_reason = ""
                    logger.info("YOLO detection statistics reset")
                elif key == ord('f'):
                    # Toggle detection speed in continuous mode
                    if self.detection_interval <= 0.1:
                        self.detection_interval = 0.3  # Normal speed
                    elif self.detection_interval <= 0.3:
                        self.detection_interval = 0.5  # Slow speed
                    elif self.detection_interval <= 0.5:
                        self.detection_interval = 1.0  # Very slow
                    else:
                        self.detection_interval = 0.1  # Fast speed
                    
                    speed_name = {0.1: "FAST", 0.3: "NORMAL", 0.5: "SLOW", 1.0: "VERY SLOW"}[self.detection_interval]
                    logger.info(f"Detection interval set to {self.detection_interval}s ({speed_name})")
                elif key == ord('s'):
                    filename = f"yolo_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    # Save the current display frame if available
                    save_frame = self.last_display_frame if self.last_display_frame is not None else frame
                    cv2.imwrite(filename, save_frame)
                    logger.info(f"Frame saved as {filename}")
                
                # Continuous detection with controlled timing
                if self.continuous_mode:
                    current_time = time.time()
                    if current_time - self.last_detection_time >= self.detection_interval:
                        should_detect = True
                        self.last_detection_time = current_time
                
                # Perform YOLO detection
                if should_detect:
                    bottles, processed_frame, processing_time = self.detect_bottles_yolo(frame)
                    status, reason = self.analyze_yolo_detections(bottles, frame.shape)
                    
                    # Update statistics
                    self.detection_stats['total_detections'] += 1
                    self.detection_stats['bottles_detected'] += len(bottles)
                    
                    # Update current detection results
                    current_bottles = bottles
                    current_status = status
                    current_reason = reason
                    current_processing_time = processing_time
                    
                    # Log result
                    logger.info(f"YOLO Detection #{self.detection_stats['total_detections']}: "
                              f"{status} - {reason} ({processing_time:.1f}ms)")
                    
                    # Save result if manual detection
                    if not self.continuous_mode:
                        self.save_yolo_detection_result(bottles, status, reason, processing_time)
                
                # Always draw overlay with current results (smooth display)
                if current_bottles or current_status:
                    # We have detection results to display
                    display_frame = self.draw_yolo_overlay(
                        frame, current_bottles, current_status, current_reason, current_processing_time
                    )
                else:
                    # Show ready state
                    display_frame = self.draw_ready_state(frame)
                
                # Store the display frame for saving
                self.last_display_frame = display_frame.copy()
                
                # Display frame
                cv2.imshow('YOLO Bottle Detection System', display_frame)
                
        except KeyboardInterrupt:
            logger.info("YOLO detection stopped by user")
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
        finally:
            self.cleanup()
    
    def draw_ready_state(self, frame):
        """Draw ready state overlay"""
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center crosshairs
        cv2.line(display_frame, (w//2-40, h//2), (w//2+40, h//2), (0, 255, 255), 3)
        cv2.line(display_frame, (w//2, h//2-40), (w//2, h//2+40), (0, 255, 255), 3)
        cv2.circle(display_frame, (w//2, h//2), self.center_tolerance, (0, 255, 255), 2)
        
        # Background for status
        cv2.rectangle(display_frame, (5, 5), (w-5, 130), (0, 0, 0), -1)
        
        cv2.putText(display_frame, "YOLO BOTTLE DETECTION READY", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Model: {self.model_name} | Confidence: {self.confidence_threshold}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show mode and speed
        mode_text = "CONTINUOUS MODE" if self.continuous_mode else "MANUAL MODE"
        mode_color = (0, 255, 0) if self.continuous_mode else (255, 255, 255)
        cv2.putText(display_frame, f"Mode: {mode_text}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        if self.continuous_mode:
            speed_name = {0.1: "FAST", 0.3: "NORMAL", 0.5: "SLOW", 1.0: "VERY SLOW"}.get(self.detection_interval, "CUSTOM")
            cv2.putText(display_frame, f"Speed: {speed_name} ({self.detection_interval}s)", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Stats
        cv2.putText(display_frame, f"Total Detections: {self.detection_stats['total_detections']} | Bottles Found: {self.detection_stats['bottles_detected']}", 
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Controls
        cv2.putText(display_frame, "SPACE:Detect C:Continuous F:Speed M:Model T:Threshold S:Save R:Reset Q:Quit", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return display_frame
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        logger.info("=" * 60)
        logger.info("YOLO BOTTLE DETECTION SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model used: {self.model_name}")
        logger.info(f"Total detections: {self.detection_stats['total_detections']}")
        logger.info(f"Bottles detected: {self.detection_stats['bottles_detected']}")
        if self.detection_stats['total_detections'] > 0:
            avg_bottles = self.detection_stats['bottles_detected'] / self.detection_stats['total_detections']
            logger.info(f"Average bottles per detection: {avg_bottles:.2f}")

def main():
    """Main function"""
    print("YOLO Bottle Detection System")
    print("=" * 70)
    print("ระบบตรวจจับขวดด้วย YOLO Deep Learning")
    print("=" * 70)
    print("Features:")
    print("- YOLOv8 real-time object detection")
    print("- Pre-trained models (nano/small/medium/large)")
    print("- High accuracy bottle detection")
    print("- Confidence scoring and filtering")
    print("- Multi-class detection support")
    print("- Real-time performance optimization")
    print("=" * 70)
    print("Available Models:")
    print("- YOLOv8n (Nano) - Fastest")
    print("- YOLOv8s (Small) - Balanced")
    print("- YOLOv8m (Medium) - Better accuracy")
    print("- YOLOv8l (Large) - Highest accuracy")
    print("=" * 70)
    print("Controls:")
    print("- SPACE: Detect bottles")
    print("- C: Toggle continuous detection")
    print("- F: Change detection speed (Fast/Normal/Slow/Very Slow)")
    print("- M: Switch YOLO model")
    print("- T: Toggle confidence threshold")
    print("- S: Save frame")
    print("- R: Reset statistics")
    print("- Q: Quit")
    print("=" * 70)
    
    if not YOLO_AVAILABLE:
        print("ERROR: YOLO not available!")
        print("Please install required packages:")
        print("pip install ultralytics torch torchvision")
        return
    
    detector = None
    try:
        detector = YOLOBottleDetector(camera_id=0)
        detector.run()
    except RuntimeError as e:
        print(f"Camera Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check camera connection")
        print("2. Try different camera_id (0, 1, 2...)")
        print("3. Close other camera applications")
        print("4. Check camera permissions")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if detector:
            detector.cleanup()

if __name__ == "__main__":
    main()