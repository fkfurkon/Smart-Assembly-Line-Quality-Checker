#!/usr/bin/env python3
"""
Bottle Detection Performance Comparison
========================================
Compare YOLO vs OpenCV bottle detection methods.

Features:
- Side-by-side performance comparison
- Speed and accuracy metrics
- Confidence scoring analysis
- Memory usage monitoring
- False positive/negative tracking

Author: Cooperative Education Project
Date: September 2025
"""

import cv2
import numpy as np
import time
import json
import logging
import psutil
import threading
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

class PerformanceComparer:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.yolo_model = None
        
        # Performance metrics
        self.metrics = {
            'yolo': {
                'total_detections': 0,
                'bottles_found': 0,
                'avg_processing_time': 0,
                'avg_confidence': 0,
                'false_positives': 0,
                'memory_usage': 0
            },
            'opencv': {
                'total_detections': 0,
                'bottles_found': 0,
                'avg_processing_time': 0,
                'avg_confidence': 0,
                'false_positives': 0,
                'memory_usage': 0
            }
        }
        
        self.init_camera()
        if YOLO_AVAILABLE:
            self.init_yolo()
    
    def init_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def init_yolo(self):
        """Initialize YOLO model"""
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("YOLO model loaded for comparison")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
    
    def detect_yolo(self, frame):
        """YOLO detection method"""
        start_time = time.time()
        
        if self.yolo_model is None:
            return [], 0, 0
        
        results = self.yolo_model(frame, conf=0.5, verbose=False)
        bottles = []
        confidences = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter for bottle classes
                    if class_id in [39, 40, 41]:  # bottle, wine glass, cup
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bottles.append({
                            'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            'confidence': confidence,
                            'method': 'yolo'
                        })
                        confidences.append(confidence)
        
        processing_time = (time.time() - start_time) * 1000
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return bottles, processing_time, avg_confidence
    
    def detect_opencv(self, frame):
        """OpenCV detection method (simplified)"""
        start_time = time.time()
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define bottle-like color ranges (blue labels, transparent bottles)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bottles = []
        confidences = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 2000:  # Minimum area for bottle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Calculate confidence based on shape characteristics
                confidence = 0.3  # Base confidence
                
                # Aspect ratio bonus (bottles are tall)
                if 1.5 <= aspect_ratio <= 4.0:
                    confidence += 0.3
                
                # Area bonus
                if 3000 <= area <= 30000:
                    confidence += 0.2
                
                # Edge detection bonus
                gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (w * h)
                if 0.1 <= edge_density <= 0.4:
                    confidence += 0.2
                
                bottles.append({
                    'bbox': (x, y, w, h),
                    'confidence': min(confidence, 1.0),
                    'method': 'opencv'
                })
                confidences.append(confidence)
        
        processing_time = (time.time() - start_time) * 1000
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return bottles, processing_time, avg_confidence
    
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def update_metrics(self, method, bottles, processing_time, avg_confidence):
        """Update performance metrics"""
        metrics = self.metrics[method]
        
        metrics['total_detections'] += 1
        metrics['bottles_found'] += len(bottles)
        
        # Update average processing time
        total_time = metrics['avg_processing_time'] * (metrics['total_detections'] - 1)
        metrics['avg_processing_time'] = (total_time + processing_time) / metrics['total_detections']
        
        # Update average confidence
        if avg_confidence > 0:
            total_conf = metrics['avg_confidence'] * max(1, metrics['bottles_found'] - len(bottles))
            total_bottles = max(1, metrics['bottles_found'])
            metrics['avg_confidence'] = (total_conf + avg_confidence * len(bottles)) / total_bottles
        
        metrics['memory_usage'] = self.get_memory_usage()
    
    def draw_comparison_overlay(self, frame, yolo_bottles, opencv_bottles, yolo_time, opencv_time):
        """Draw comparison overlay"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Split screen visualization
        split_x = w // 2
        
        # Draw dividing line
        cv2.line(overlay, (split_x, 0), (split_x, h), (255, 255, 255), 2)
        
        # YOLO side (left)
        cv2.putText(overlay, "YOLO Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        for bottle in yolo_bottles:
            x, y, w, h = bottle['bbox']
            if x < split_x:  # Only draw on left side
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(overlay, f"YOLO {bottle['confidence']:.2f}", 
                           (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # OpenCV side (right)
        cv2.putText(overlay, "OpenCV Detection", (split_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        for bottle in opencv_bottles:
            x, y, w, h = bottle['bbox']
            if x > split_x:  # Only draw on right side
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(overlay, f"CV {bottle['confidence']:.2f}", 
                           (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Performance metrics
        y_offset = 60
        cv2.putText(overlay, f"YOLO: {len(yolo_bottles)} bottles, {yolo_time:.1f}ms", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(overlay, f"OpenCV: {len(opencv_bottles)} bottles, {opencv_time:.1f}ms", 
                   (split_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Overall statistics
        y_stats = h - 100
        cv2.rectangle(overlay, (0, y_stats), (w, h), (0, 0, 0), -1)
        
        yolo_metrics = self.metrics['yolo']
        opencv_metrics = self.metrics['opencv']
        
        # YOLO stats
        cv2.putText(overlay, "YOLO Statistics:", (10, y_stats + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f"Avg Time: {yolo_metrics['avg_processing_time']:.1f}ms", 
                   (10, y_stats + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"Avg Conf: {yolo_metrics['avg_confidence']:.2f}", 
                   (10, y_stats + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"Total: {yolo_metrics['bottles_found']}/{yolo_metrics['total_detections']}", 
                   (10, y_stats + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # OpenCV stats
        cv2.putText(overlay, "OpenCV Statistics:", (split_x + 10, y_stats + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(overlay, f"Avg Time: {opencv_metrics['avg_processing_time']:.1f}ms", 
                   (split_x + 10, y_stats + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"Avg Conf: {opencv_metrics['avg_confidence']:.2f}", 
                   (split_x + 10, y_stats + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"Total: {opencv_metrics['bottles_found']}/{opencv_metrics['total_detections']}", 
                   (split_x + 10, y_stats + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return overlay
    
    def save_comparison_results(self):
        """Save comparison results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'comparison_results': self.metrics,
            'winner': self.determine_winner()
        }
        
        filename = f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comparison results saved to {filename}")
        return filename
    
    def determine_winner(self):
        """Determine which method performs better"""
        yolo_score = 0
        opencv_score = 0
        
        yolo_metrics = self.metrics['yolo']
        opencv_metrics = self.metrics['opencv']
        
        # Speed comparison (lower is better)
        if yolo_metrics['avg_processing_time'] < opencv_metrics['avg_processing_time']:
            yolo_score += 1
        else:
            opencv_score += 1
        
        # Confidence comparison (higher is better)
        if yolo_metrics['avg_confidence'] > opencv_metrics['avg_confidence']:
            yolo_score += 1
        else:
            opencv_score += 1
        
        # Detection rate comparison
        yolo_rate = yolo_metrics['bottles_found'] / max(1, yolo_metrics['total_detections'])
        opencv_rate = opencv_metrics['bottles_found'] / max(1, opencv_metrics['total_detections'])
        
        if yolo_rate > opencv_rate:
            yolo_score += 1
        else:
            opencv_score += 1
        
        if yolo_score > opencv_score:
            return "YOLO"
        elif opencv_score > yolo_score:
            return "OpenCV"
        else:
            return "Tie"
    
    def run_comparison(self):
        """Run performance comparison"""
        if not YOLO_AVAILABLE:
            print("YOLO not available. Cannot run comparison.")
            return
        
        print("Bottle Detection Performance Comparison")
        print("=" * 60)
        print("Left side: YOLO Detection")
        print("Right side: OpenCV Detection")
        print("=" * 60)
        print("Controls:")
        print("- SPACE: Run comparison")
        print("- C: Continuous comparison")
        print("- S: Save results")
        print("- R: Reset statistics")
        print("- Q: Quit")
        print("=" * 60)
        
        continuous_mode = False
        last_comparison_time = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                split_x = w // 2
                
                # Create split view for detection
                left_frame = frame[:, :split_x].copy()
                right_frame = frame[:, split_x:].copy()
                
                # Resize frames back to full width for detection
                left_frame_full = cv2.resize(left_frame, (w, h))
                right_frame_full = cv2.resize(right_frame, (w, h))
                
                key = cv2.waitKey(1) & 0xFF
                should_compare = False
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    should_compare = True
                elif key == ord('c'):
                    continuous_mode = not continuous_mode
                    logger.info(f"Continuous comparison: {'ON' if continuous_mode else 'OFF'}")
                elif key == ord('s'):
                    self.save_comparison_results()
                elif key == ord('r'):
                    self.metrics = {
                        'yolo': {'total_detections': 0, 'bottles_found': 0, 'avg_processing_time': 0, 
                                'avg_confidence': 0, 'false_positives': 0, 'memory_usage': 0},
                        'opencv': {'total_detections': 0, 'bottles_found': 0, 'avg_processing_time': 0, 
                                  'avg_confidence': 0, 'false_positives': 0, 'memory_usage': 0}
                    }
                    logger.info("Statistics reset")
                
                # Continuous mode
                if continuous_mode:
                    current_time = time.time()
                    if current_time - last_comparison_time > 1.0:  # Every 1 second
                        should_compare = True
                        last_comparison_time = current_time
                
                if should_compare:
                    # Run both detection methods
                    yolo_bottles, yolo_time, yolo_conf = self.detect_yolo(left_frame_full)
                    opencv_bottles, opencv_time, opencv_conf = self.detect_opencv(right_frame_full)
                    
                    # Adjust coordinates for display
                    for bottle in opencv_bottles:
                        x, y, w, h = bottle['bbox']
                        bottle['bbox'] = (x + split_x, y, w, h)
                    
                    # Update metrics
                    self.update_metrics('yolo', yolo_bottles, yolo_time, yolo_conf)
                    self.update_metrics('opencv', opencv_bottles, opencv_time, opencv_conf)
                    
                    # Draw comparison
                    display_frame = self.draw_comparison_overlay(
                        frame, yolo_bottles, opencv_bottles, yolo_time, opencv_time
                    )
                    
                    logger.info(f"Comparison: YOLO {len(yolo_bottles)} bottles ({yolo_time:.1f}ms) "
                               f"vs OpenCV {len(opencv_bottles)} bottles ({opencv_time:.1f}ms)")
                else:
                    # Show ready state
                    display_frame = frame.copy()
                    cv2.line(display_frame, (split_x, 0), (split_x, h), (255, 255, 255), 2)
                    cv2.putText(display_frame, "YOLO vs OpenCV Comparison Ready", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Press SPACE to compare methods", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Bottle Detection Comparison', display_frame)
                
        except KeyboardInterrupt:
            logger.info("Comparison stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final comparison
        print("\n" + "=" * 60)
        print("FINAL COMPARISON RESULTS")
        print("=" * 60)
        
        yolo_metrics = self.metrics['yolo']
        opencv_metrics = self.metrics['opencv']
        
        print(f"YOLO Performance:")
        print(f"  Average processing time: {yolo_metrics['avg_processing_time']:.1f}ms")
        print(f"  Average confidence: {yolo_metrics['avg_confidence']:.2f}")
        print(f"  Detection rate: {yolo_metrics['bottles_found']}/{yolo_metrics['total_detections']}")
        print(f"  Memory usage: {yolo_metrics['memory_usage']:.1f}MB")
        
        print(f"\nOpenCV Performance:")
        print(f"  Average processing time: {opencv_metrics['avg_processing_time']:.1f}ms")
        print(f"  Average confidence: {opencv_metrics['avg_confidence']:.2f}")
        print(f"  Detection rate: {opencv_metrics['bottles_found']}/{opencv_metrics['total_detections']}")
        print(f"  Memory usage: {opencv_metrics['memory_usage']:.1f}MB")
        
        winner = self.determine_winner()
        print(f"\nWinner: {winner}")
        print("=" * 60)

def main():
    comparer = PerformanceComparer(camera_id=0)
    comparer.run_comparison()

if __name__ == "__main__":
    main()