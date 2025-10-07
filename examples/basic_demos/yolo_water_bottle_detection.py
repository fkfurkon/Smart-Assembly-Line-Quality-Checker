#!/usr/bin/env python3
"""
YOLO Water Bottle Specific Detection
====================================
Specialized YOLO detection optimized for water bottles, especially Nestlé Pure Life.
Combines YOLO object detection with additional filtering for water bottle characteristics.

Features:
- YOLO pre-trained model for initial detection
- Water bottle specific post-processing
- Label detection for branded bottles
- Transparent bottle edge enhancement
- Confidence scoring based on bottle characteristics
- Real-time performance optimization

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Install with: pip install ultralytics")

class YOLOWaterBottleDetector:
    def __init__(self, camera_id=0):
        """Initialize specialized water bottle detector"""
        self.camera_id = camera_id
        self.cap = None
        self.yolo_model = None
        
        # Water bottle specific parameters
        self.min_bottle_area = 3000
        self.max_bottle_area = 50000
        self.min_aspect_ratio = 1.8
        self.max_aspect_ratio = 4.0
        self.confidence_threshold = 0.4  # Lower for initial YOLO detection
        self.final_confidence_threshold = 0.7  # Higher for final output
        
        # Nestlé specific color ranges (HSV)
        self.nestle_blue_lower = np.array([100, 40, 40])
        self.nestle_blue_upper = np.array([130, 255, 255])
        self.label_white_lower = np.array([0, 0, 180])
        self.label_white_upper = np.array([180, 30, 255])
        
        # Detection statistics
        self.stats = {
            'total_scans': 0,
            'yolo_detections': 0,
            'water_bottles_confirmed': 0,
            'nestle_bottles_detected': 0,
            'avg_processing_time': 0,
            'session_start': datetime.now().isoformat()
        }
        
        self.continuous_mode = False
        self.last_detection_time = 0
        
        self.init_camera()
        if YOLO_AVAILABLE:
            self.init_yolo()
    
    def init_camera(self):
        """Initialize camera connection"""
        logger.info(f"Initializing camera {self.camera_id} for water bottle detection...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            ret, frame = self.cap.read()
            if ret and frame is not None:
                logger.info(f"Camera ready for water bottle detection")
                logger.info(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return
        
        raise RuntimeError("Camera initialization failed")
    
    def init_yolo(self):
        """Initialize YOLO model"""
        try:
            logger.info("Loading YOLO model for water bottle detection...")
            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
    
    def detect_bottles_yolo(self, frame):
        """Initial YOLO detection for bottle-like objects"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(
                frame,
                conf=self.confidence_threshold,
                iou=0.45,
                verbose=False
            )
            
            bottle_candidates = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Filter for bottle/cup classes
                        class_name = self.yolo_model.names[class_id]
                        if (class_id in [39, 40, 41] or  # bottle, wine glass, cup
                            any(keyword in class_name.lower() for keyword in ['bottle', 'cup'])):
                            
                            width = int(x2 - x1)
                            height = int(y2 - y1)
                            area = width * height
                            aspect_ratio = height / width if width > 0 else 0
                            
                            # Basic size and shape filtering
                            if (self.min_bottle_area <= area <= self.max_bottle_area and
                                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                                
                                bottle_candidates.append({
                                    'class_id': class_id,
                                    'class_name': class_name,
                                    'yolo_confidence': confidence,
                                    'bbox': (int(x1), int(y1), width, height),
                                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'coordinates': (x1, y1, x2, y2)
                                })
            
            return bottle_candidates
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def analyze_water_bottle_characteristics(self, frame, bottle_candidate):
        """Analyze specific water bottle characteristics"""
        x, y, w, h = bottle_candidate['bbox']
        
        # Extract bottle region
        bottle_roi = frame[y:y+h, x:x+w]
        if bottle_roi.size == 0:
            return 0.0, {}
        
        # Convert to HSV for color analysis
        hsv_roi = cv2.cvtColor(bottle_roi, cv2.COLOR_BGR2HSV)
        
        characteristics = {
            'nestle_blue_detected': False,
            'white_label_detected': False,
            'transparent_bottle': False,
            'neck_narrowing': False,
            'cylindrical_shape': False
        }
        
        confidence_score = bottle_candidate['yolo_confidence'] * 0.3  # Base from YOLO
        
        # 1. Check for Nestlé blue label
        blue_mask = cv2.inRange(hsv_roi, self.nestle_blue_lower, self.nestle_blue_upper)
        blue_pixels = np.sum(blue_mask > 0)
        blue_ratio = blue_pixels / (w * h)
        
        if blue_ratio > 0.05:  # At least 5% blue pixels
            characteristics['nestle_blue_detected'] = True
            confidence_score += 0.25
        
        # 2. Check for white label areas
        white_mask = cv2.inRange(hsv_roi, self.label_white_lower, self.label_white_upper)
        white_pixels = np.sum(white_mask > 0)
        white_ratio = white_pixels / (w * h)
        
        if white_ratio > 0.15:  # At least 15% white pixels (label area)
            characteristics['white_label_detected'] = True
            confidence_score += 0.2
        
        # 3. Check for transparent bottle characteristics (edge detection)
        gray_roi = cv2.cvtColor(bottle_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 30, 80)
        edge_pixels = np.sum(edges > 0)
        edge_density = edge_pixels / (w * h)
        
        if 0.08 <= edge_density <= 0.25:  # Moderate edge density for transparent bottles
            characteristics['transparent_bottle'] = True
            confidence_score += 0.15
        
        # 4. Check for neck narrowing (top 1/3 should be narrower)
        top_third = bottle_roi[:h//3, :]
        middle_third = bottle_roi[h//3:2*h//3, :]
        
        if top_third.size > 0 and middle_third.size > 0:
            top_edges = cv2.Canny(cv2.cvtColor(top_third, cv2.COLOR_BGR2GRAY), 30, 80)
            middle_edges = cv2.Canny(cv2.cvtColor(middle_third, cv2.COLOR_BGR2GRAY), 30, 80)
            
            top_width = np.sum(np.any(top_edges > 0, axis=0))
            middle_width = np.sum(np.any(middle_edges > 0, axis=0))
            
            if middle_width > 0 and top_width / middle_width < 0.8:  # Top is narrower
                characteristics['neck_narrowing'] = True
                confidence_score += 0.1
        
        # 5. Check cylindrical shape consistency
        aspect_ratio = bottle_candidate['aspect_ratio']
        if 2.2 <= aspect_ratio <= 3.5:  # Ideal water bottle proportions
            characteristics['cylindrical_shape'] = True
            confidence_score += 0.1
        
        return min(confidence_score, 1.0), characteristics
    
    def filter_water_bottles(self, frame, bottle_candidates):
        """Filter and validate water bottles"""
        confirmed_bottles = []
        
        for candidate in bottle_candidates:
            confidence, characteristics = self.analyze_water_bottle_characteristics(frame, candidate)
            
            # Only accept bottles that meet minimum confidence
            if confidence >= self.final_confidence_threshold:
                candidate['final_confidence'] = confidence
                candidate['characteristics'] = characteristics
                
                # Determine bottle type
                if characteristics['nestle_blue_detected'] and characteristics['white_label_detected']:
                    candidate['bottle_type'] = 'Nestlé Pure Life'
                    candidate['brand_confidence'] = 0.9
                elif characteristics['nestle_blue_detected']:
                    candidate['bottle_type'] = 'Nestlé Brand'
                    candidate['brand_confidence'] = 0.7
                elif characteristics['transparent_bottle'] and characteristics['white_label_detected']:
                    candidate['bottle_type'] = 'Branded Water Bottle'
                    candidate['brand_confidence'] = 0.6
                else:
                    candidate['bottle_type'] = 'Generic Water Bottle'
                    candidate['brand_confidence'] = 0.4
                
                confirmed_bottles.append(candidate)
        
        # Sort by confidence
        confirmed_bottles.sort(key=lambda x: x['final_confidence'], reverse=True)
        return confirmed_bottles
    
    def analyze_detection_result(self, bottles, frame_shape):
        """Analyze final detection result"""
        if len(bottles) == 0:
            return "FAIL", "ไม่พบขวดน้ำ - No water bottles detected"
        
        # Check for multiple bottles
        if len(bottles) > 1:
            high_confidence_bottles = [b for b in bottles if b['final_confidence'] > 0.8]
            if len(high_confidence_bottles) > 1:
                return "FAIL", f"พบขวดหลายใบ - Multiple bottles detected ({len(bottles)})"
        
        # Analyze best bottle
        best_bottle = bottles[0]
        bottle_type = best_bottle['bottle_type']
        confidence = best_bottle['final_confidence']
        characteristics = best_bottle['characteristics']
        
        # Check position (should be reasonably centered)
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        bottle_center_x, bottle_center_y = best_bottle['center']
        
        distance_from_center = np.sqrt(
            (bottle_center_x - frame_center_x)**2 + 
            (bottle_center_y - frame_center_y)**2
        )
        
        if distance_from_center > 150:  # More lenient than before
            return "FAIL", f"ขวดไม่อยู่กึ่งกลาง - Off-center ({distance_from_center:.0f}px)"
        
        # Success message with details
        char_list = []
        if characteristics['nestle_blue_detected']:
            char_list.append("Blue Label")
        if characteristics['white_label_detected']:
            char_list.append("White Label")
        if characteristics['transparent_bottle']:
            char_list.append("Transparent")
        if characteristics['neck_narrowing']:
            char_list.append("Neck")
        
        char_str = ", ".join(char_list) if char_list else "Basic"
        
        return "PASS", f"ตรวจพบ {bottle_type} (conf: {confidence:.2f}, {char_str})"
    
    def draw_overlay(self, frame, bottles, status, reason, processing_time):
        """Draw detection overlay"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center crosshairs
        cv2.line(overlay, (w//2-50, h//2), (w//2+50, h//2), (0, 255, 255), 2)
        cv2.line(overlay, (w//2, h//2-50), (w//2, h//2+50), (0, 255, 255), 2)
        cv2.circle(overlay, (w//2, h//2), 150, (0, 255, 255), 1)
        
        # Draw detected bottles
        for i, bottle in enumerate(bottles):
            x, y, w_box, h_box = bottle['bbox']
            confidence = bottle['final_confidence']
            bottle_type = bottle['bottle_type']
            characteristics = bottle['characteristics']
            
            # Color based on bottle type
            if 'Nestlé' in bottle_type:
                color = (0, 255, 0)  # Green for Nestlé
            elif 'Branded' in bottle_type:
                color = (0, 255, 255)  # Yellow for other brands
            else:
                color = (255, 0, 0)  # Blue for generic
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x+w_box, y+h_box), color, 2)
            
            # Draw center point
            center = bottle['center']
            cv2.circle(overlay, center, 8, color, -1)
            
            # Draw label with bottle type
            label_bg_height = 60
            cv2.rectangle(overlay, (x, y-label_bg_height), (x+300, y), color, -1)
            
            cv2.putText(overlay, bottle_type, (x+5, y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(overlay, f"Confidence: {confidence:.2f}", (x+5, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(overlay, f"YOLO: {bottle['yolo_confidence']:.2f}", (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Draw characteristics indicators
            y_char = y + h_box + 20
            if characteristics['nestle_blue_detected']:
                cv2.putText(overlay, "BLUE", (x, y_char), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                x += 50
            if characteristics['white_label_detected']:
                cv2.putText(overlay, "LABEL", (x, y_char), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                x += 60
            if characteristics['transparent_bottle']:
                cv2.putText(overlay, "CLEAR", (x, y_char), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Status display
        status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        
        # Background for status
        cv2.rectangle(overlay, (5, 5), (w-5, 120), (0, 0, 0), -1)
        
        cv2.putText(overlay, f"YOLO WATER BOTTLE DETECTION: {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(overlay, reason, 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(overlay, f"Processing: {processing_time:.1f}ms | Bottles: {len(bottles)}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"YOLO Conf: {self.confidence_threshold} | Final: {self.final_confidence_threshold}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Statistics
        mode_text = "CONTINUOUS" if self.continuous_mode else "MANUAL"
        cv2.putText(overlay, f"Mode: {mode_text} | Scans: {self.stats['total_scans']} | Confirmed: {self.stats['water_bottles_confirmed']}", 
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Controls
        cv2.putText(overlay, "SPACE:Detect C:Continuous S:Save T:Threshold R:Reset Q:Quit", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return overlay
    
    def save_detection_result(self, bottles, status, reason, processing_time):
        """Save detection results"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'detection_type': 'yolo_water_bottle_detection',
            'status': status,
            'reason': reason,
            'processing_time_ms': processing_time,
            'yolo_confidence_threshold': self.confidence_threshold,
            'final_confidence_threshold': self.final_confidence_threshold,
            'bottles_detected': len(bottles),
            'bottles': []
        }
        
        for bottle in bottles:
            bottle_data = {
                'bottle_type': bottle['bottle_type'],
                'brand_confidence': bottle['brand_confidence'],
                'yolo_confidence': float(bottle['yolo_confidence']),
                'final_confidence': float(bottle['final_confidence']),
                'center': bottle['center'],
                'bbox': bottle['bbox'],
                'area': bottle['area'],
                'aspect_ratio': float(bottle['aspect_ratio']),
                'characteristics': bottle['characteristics']
            }
            result['bottles'].append(bottle_data)
        
        filename = f"yolo_water_bottle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Water bottle detection result saved to {filename}")
        return filename
    
    def run(self):
        """Main detection loop"""
        if not YOLO_AVAILABLE:
            print("YOLO is not available. Please install ultralytics:")
            print("pip install ultralytics")
            return
        
        logger.info("Starting YOLO water bottle detection...")
        logger.info(f"YOLO confidence threshold: {self.confidence_threshold}")
        logger.info(f"Final confidence threshold: {self.final_confidence_threshold}")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                should_detect = False
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    should_detect = True
                elif key == ord('c'):
                    self.continuous_mode = not self.continuous_mode
                    logger.info(f"Continuous detection: {'ON' if self.continuous_mode else 'OFF'}")
                elif key == ord('t'):
                    # Toggle confidence thresholds
                    if self.final_confidence_threshold >= 0.7:
                        self.final_confidence_threshold = 0.5
                        self.confidence_threshold = 0.3
                    else:
                        self.final_confidence_threshold = 0.7
                        self.confidence_threshold = 0.4
                    logger.info(f"Thresholds: YOLO={self.confidence_threshold}, Final={self.final_confidence_threshold}")
                elif key == ord('r'):
                    self.stats = {
                        'total_scans': 0,
                        'yolo_detections': 0,
                        'water_bottles_confirmed': 0,
                        'nestle_bottles_detected': 0,
                        'avg_processing_time': 0,
                        'session_start': datetime.now().isoformat()
                    }
                    logger.info("Statistics reset")
                elif key == ord('s'):
                    filename = f"water_bottle_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved as {filename}")
                
                # Continuous detection
                if self.continuous_mode:
                    current_time = time.time()
                    if current_time - self.last_detection_time > 0.5:
                        should_detect = True
                        self.last_detection_time = current_time
                
                # Perform detection
                if should_detect:
                    start_time = time.time()
                    
                    # YOLO detection
                    bottle_candidates = self.detect_bottles_yolo(frame)
                    
                    # Water bottle filtering
                    confirmed_bottles = self.filter_water_bottles(frame, bottle_candidates)
                    
                    # Analysis
                    status, reason = self.analyze_detection_result(confirmed_bottles, frame.shape)
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Update statistics
                    self.stats['total_scans'] += 1
                    self.stats['yolo_detections'] += len(bottle_candidates)
                    self.stats['water_bottles_confirmed'] += len(confirmed_bottles)
                    
                    # Count Nestlé bottles
                    nestle_count = sum(1 for b in confirmed_bottles if 'Nestlé' in b['bottle_type'])
                    self.stats['nestle_bottles_detected'] += nestle_count
                    
                    # Update average processing time
                    total_time = self.stats['avg_processing_time'] * (self.stats['total_scans'] - 1)
                    self.stats['avg_processing_time'] = (total_time + processing_time) / self.stats['total_scans']
                    
                    # Draw overlay
                    display_frame = self.draw_overlay(frame, confirmed_bottles, status, reason, processing_time)
                    
                    # Log result
                    logger.info(f"Scan #{self.stats['total_scans']}: {status} - {reason} "
                               f"({processing_time:.1f}ms, YOLO: {len(bottle_candidates)}, Confirmed: {len(confirmed_bottles)})")
                    
                    # Save if manual detection
                    if not self.continuous_mode:
                        self.save_detection_result(confirmed_bottles, status, reason, processing_time)
                
                else:
                    # Ready state
                    h, w = frame.shape[:2]
                    cv2.putText(display_frame, "YOLO WATER BOTTLE DETECTION READY", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Optimized for Nestlé Pure Life bottles", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(display_frame, "SPACE:Detect C:Continuous T:Threshold S:Save Q:Quit", 
                               (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.imshow('YOLO Water Bottle Detection', display_frame)
        
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        except Exception as e:
            logger.error(f"Detection error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        logger.info("=" * 70)
        logger.info("YOLO WATER BOTTLE DETECTION SESSION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total scans: {self.stats['total_scans']}")
        logger.info(f"YOLO detections: {self.stats['yolo_detections']}")
        logger.info(f"Water bottles confirmed: {self.stats['water_bottles_confirmed']}")
        logger.info(f"Nestlé bottles detected: {self.stats['nestle_bottles_detected']}")
        logger.info(f"Average processing time: {self.stats['avg_processing_time']:.1f}ms")
        
        if self.stats['total_scans'] > 0:
            confirmation_rate = self.stats['water_bottles_confirmed'] / self.stats['total_scans'] * 100
            logger.info(f"Confirmation rate: {confirmation_rate:.1f}%")

def main():
    print("YOLO Water Bottle Detection System")
    print("=" * 70)
    print("Specialized detection for water bottles")
    print("Optimized for Nestlé Pure Life bottles")
    print("=" * 70)
    print("Features:")
    print("- YOLO object detection + water bottle filtering")
    print("- Nestlé brand recognition")
    print("- Transparent bottle detection")
    print("- Label color analysis")
    print("- Shape and proportion validation")
    print("=" * 70)
    
    if not YOLO_AVAILABLE:
        print("ERROR: YOLO not available!")
        print("Please install: pip install ultralytics")
        return
    
    try:
        detector = YOLOWaterBottleDetector(camera_id=0)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()