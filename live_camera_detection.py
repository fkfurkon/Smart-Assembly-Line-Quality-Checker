#!/usr/bin/env python3
"""
Live Bottle Detection System
============================
Specialized real-time bottle detection using computer vision for assembly line quality control.
Optimized to detect bottles based on shape characteristics like aspect ratio, circularity, and size.

Features:
- Bottle shape analysis (height/width ratio, circularity)
- Size filtering for different bottle types
- Position centering validation for quality control
- Real-time processing with detailed feedback
- Industrial assembly line integration ready

Controls:
- SPACE: Capture and analyze frame for bottles
- S: Save current frame
- C: Toggle continuous detection mode
- R: Reset detection statistics
- Q: Quit

Detection Criteria:
- Area: 2000-80000 pixels (adjustable for bottle size)
- Aspect Ratio: 1.5-4.0 (bottles are taller than wide)
- Circularity: 0.1-0.8 (moderate roundness)
- Solidity: >0.6 (mostly solid shape)
- Position: Within center tolerance for quality control

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveCameraDetector:
    def __init__(self, camera_id=0):
        """Initialize camera detector"""
        self.camera_id = camera_id
        self.cap = None
        self.detection_stats = {
            'total_detections': 0,
            'objects_detected': 0,
            'session_start': datetime.now().isoformat()
        }
        self.continuous_mode = False
        self.last_detection_time = 0
        
        # Detection parameters optimized for water bottles (like Nestlé Pure Life)
        self.min_area = 4000      # Water bottles are medium-large sized
        self.max_area = 45000     # Not too large for typical water bottles
        self.center_tolerance = 100  # Stricter centering for quality control
        self.min_aspect_ratio = 2.2  # Water bottles are tall and narrow
        self.max_aspect_ratio = 3.8  # Allow for various bottle heights
        
        # Advanced shape analysis parameters for transparent bottles
        self.min_circularity = 0.20  # Bottles with labels have moderate circularity
        self.max_circularity = 0.70  # Not perfectly round
        self.min_solidity = 0.75     # Should be mostly solid despite transparency
        self.min_extent = 0.65       # Bottle should fill most of bounding box
        
        # Edge detection parameters tuned for transparent bottles
        self.canny_low = 30      # Lower threshold for transparent edges
        self.canny_high = 100    # Adjusted for bottle edges
        
        # Color analysis for label detection
        self.detect_labels = True  # Enable label detection for better accuracy
        
        # Initialize camera
        self.init_camera()
        
    def init_camera(self):
        """Initialize camera connection"""
        logger.info(f"Initializing camera {self.camera_id}...")
        
        # Try different camera backends
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                if self.cap.isOpened():
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Test frame capture
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info(f"Camera initialized successfully with backend {backend}")
                        logger.info(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                        return
                    
            except Exception as e:
                logger.warning(f"Failed to initialize camera with backend {backend}: {e}")
                continue
                
        # If we get here, camera initialization failed
        logger.error("Failed to initialize camera!")
        logger.info("Available cameras:")
        self.list_available_cameras()
        raise RuntimeError("Camera initialization failed")
        
    def list_available_cameras(self):
        """List available camera devices"""
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    logger.info(f"  Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
                else:
                    logger.info(f"  Camera {i}: Detected but no frame")
                cap.release()
            
    def detect_bottles(self, frame):
        """Enhanced bottle detection optimized for transparent water bottles with labels"""
        start_time = time.time()
        
        # Convert to different color spaces for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Method 1: Enhanced edge detection for bottle outlines (especially transparent ones)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Method 2: Adaptive thresholding for varying lighting
        thresh_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 19, 4
        )
        
        # Method 3: Color-based detection for bottle labels (blue label like Nestlé)
        label_mask = None
        if self.detect_labels:
            # Define blue color range for typical water bottle labels
            lower_blue = np.array([100, 50, 50])   # Lower HSV for blue
            upper_blue = np.array([130, 255, 255]) # Upper HSV for blue
            label_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Also detect other common label colors
            lower_cyan = np.array([80, 50, 50])
            upper_cyan = np.array([100, 255, 255])
            cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
            
            label_mask = cv2.bitwise_or(label_mask, cyan_mask)
        
        # Method 4: Otsu's thresholding
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine all detection methods
        combined_thresh = cv2.bitwise_or(thresh_adaptive, edges)
        combined_thresh = cv2.bitwise_or(combined_thresh, thresh_otsu)
        
        # Add label information if available
        if label_mask is not None:
            # Dilate label mask to connect with bottle body
            label_dilated = cv2.dilate(label_mask, np.ones((15, 15), np.uint8), iterations=1)
            combined_thresh = cv2.bitwise_or(combined_thresh, label_dilated)
        
        # Enhanced morphological operations for bottle shapes
        # Vertical kernel to connect bottle parts (bottles are tall)
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Connect vertical parts of bottles first
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_vertical)
        # Then general closing to fill gaps
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        # Remove small noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Enhanced bottle filtering specifically for water bottles
        valid_bottles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Basic area filter for water bottles
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box and calculate properties
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
                
            aspect_ratio = h / w
            
            # Water bottles are typically tall and narrow
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Calculate advanced shape properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Circularity measure
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity or circularity > self.max_circularity:
                continue
            
            # Convex hull analysis
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
                
            solidity = area / hull_area
            if solidity < self.min_solidity:
                continue
            
            # Extent (ratio of contour area to bounding rectangle area)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            if extent < self.min_extent:
                continue
            
            # Contour approximation for bottle shape
            epsilon = 0.015 * perimeter  # Tighter approximation for bottles
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Water bottles typically have 4-10 major corners when approximated
            if len(approx) < 4 or len(approx) > 15:
                continue
            
            # Bottle neck detection (specific for water bottles)
            # Find the topmost points and check for neck narrowing
            top_points = contour[contour[:, :, 1].argsort()][:max(1, len(contour)//10)]
            bottom_points = contour[contour[:, :, 1].argsort()][-max(1, len(contour)//10):]
            
            if len(top_points) > 0 and len(bottom_points) > 0:
                top_y = np.mean(top_points[:, 0, 1])
                bottom_y = np.mean(bottom_points[:, 0, 1])
                bottle_height = bottom_y - top_y
                
                # Check if bottle uses most of its height
                if bottle_height < h * 0.75:
                    continue
                
                # Check for neck narrowing (water bottles have narrow necks)
                top_quarter = y + h * 0.25
                neck_contour = contour[contour[:, :, 1] < top_quarter]
                body_contour = contour[contour[:, :, 1] > top_quarter]
                
                if len(neck_contour) > 0 and len(body_contour) > 0:
                    neck_width = np.max(neck_contour[:, 0, 0]) - np.min(neck_contour[:, 0, 0])
                    body_width = np.max(body_contour[:, 0, 0]) - np.min(body_contour[:, 0, 0])
                    
                    # Neck should be narrower than body for water bottles
                    neck_ratio = neck_width / body_width if body_width > 0 else 1
                    if neck_ratio > 0.9:  # Neck not narrow enough
                        continue
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Check for label presence in the contour area
            label_confidence = 0
            if label_mask is not None:
                roi_mask = label_mask[y:y+h, x:x+w]
                if roi_mask.size > 0:
                    label_pixels = cv2.countNonZero(roi_mask)
                    label_confidence = (label_pixels / (w * h)) * 100
            
            # Moments for centroid calculation
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])
            else:
                centroid_x, centroid_y = center_x, center_y
            
            # Check centroid deviation
            centroid_deviation = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
            if centroid_deviation > min(w, h) * 0.25:
                continue
            
            # Calculate enhanced confidence score
            confidence = self.calculate_water_bottle_confidence(
                area, aspect_ratio, circularity, solidity, extent, label_confidence
            )
            
            valid_bottles.append({
                'contour': contour,
                'area': area,
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'solidity': solidity,
                'extent': extent,
                'approx_corners': len(approx),
                'centroid': (centroid_x, centroid_y),
                'label_confidence': label_confidence,
                'confidence': confidence
            })
        
        # Sort bottles by confidence score (highest first)
        valid_bottles.sort(key=lambda x: x['confidence'], reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        
        return valid_bottles, cleaned, processing_time
    
    def calculate_water_bottle_confidence(self, area, aspect_ratio, circularity, solidity, extent, label_confidence):
        """Calculate confidence score specifically for water bottles (0-100)"""
        confidence = 0
        
        # Area score (25% weight) - ideal for water bottles
        ideal_area = 12000  # Typical water bottle area
        area_diff = abs(area - ideal_area) / ideal_area
        area_score = max(0, 100 - area_diff * 100)
        confidence += area_score * 0.25
        
        # Aspect ratio score (30% weight) - very important for water bottles
        ideal_aspect = 2.8  # Typical water bottle ratio
        aspect_diff = abs(aspect_ratio - ideal_aspect) / ideal_aspect
        aspect_score = max(0, 100 - aspect_diff * 100)
        confidence += aspect_score * 0.30
        
        # Circularity score (15% weight)
        ideal_circularity = 0.45
        circ_diff = abs(circularity - ideal_circularity) / ideal_circularity
        circ_score = max(0, 100 - circ_diff * 100)
        confidence += circ_score * 0.15
        
        # Solidity score (15% weight)
        ideal_solidity = 0.82
        solid_diff = abs(solidity - ideal_solidity) / ideal_solidity
        solid_score = max(0, 100 - solid_diff * 100)
        confidence += solid_score * 0.15
        
        # Extent score (10% weight)
        ideal_extent = 0.75
        extent_diff = abs(extent - ideal_extent) / ideal_extent
        extent_score = max(0, 100 - extent_diff * 100)
        confidence += extent_score * 0.10
        
        # Label confidence bonus (5% weight) - bottles with labels are more likely to be real
        confidence += min(label_confidence, 100) * 0.05
        
        return confidence
        
    def analyze_bottle_detection(self, bottles, frame_shape):
        """Enhanced analysis specifically for water bottles like Nestlé Pure Life"""
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        
        if len(bottles) == 0:
            return "FAIL", "ไม่พบขวดน้ำ - No water bottles detected"
        elif len(bottles) > 1:
            # Check confidence difference for multiple bottles
            if len(bottles) >= 2 and bottles[0]['confidence'] - bottles[1]['confidence'] < 15:
                return "FAIL", f"พบขวดหลายใบ ({len(bottles)} ขวด) - Multiple bottles with similar confidence"
            else:
                # Take the highest confidence bottle
                bottles = [bottles[0]]
        
        bottle = bottles[0]
        center_x, center_y = bottle['center']
        area = bottle['area']
        aspect_ratio = bottle['aspect_ratio']
        circularity = bottle['circularity']
        solidity = bottle['solidity']
        confidence = bottle['confidence']
        label_confidence = bottle.get('label_confidence', 0)
        
        # Enhanced validation with stricter criteria for water bottles
        if confidence < 70:  # Higher threshold for water bottles
            return "FAIL", f"ความน่าเชื่อถือต่ำ - Low confidence ({confidence:.1f}%)"
        
        # Check if bottle is centered
        distance_from_center = np.sqrt(
            (center_x - frame_center_x)**2 + (center_y - frame_center_y)**2
        )
        
        if distance_from_center > self.center_tolerance:
            return "FAIL", f"ขวดไม่อยู่กึ่งกลาง - Bottle off-center ({distance_from_center:.1f}px)"
        elif area < self.min_area * 1.1:  # Area check for water bottles
            return "FAIL", f"ขวดเล็กเกินไป - Bottle too small (area: {area:.0f}px)"
        elif aspect_ratio < 2.3:  # Water bottles should be quite tall
            return "FAIL", f"รูปร่างไม่เหมือนขวดน้ำ - Not water bottle shape (ratio: {aspect_ratio:.2f})"
        elif circularity > 0.65:  # Too round to be a bottle
            return "FAIL", f"วัตถุกลมเกินไป - Object too round (circularity: {circularity:.3f})"
        elif solidity < 0.78:  # Water bottles should be quite solid
            return "FAIL", f"รูปร่างไม่สม่ำเสมอ - Irregular shape (solidity: {solidity:.3f})"
        else:
            # Successful detection with detailed info
            label_info = f", label: {label_confidence:.1f}%" if label_confidence > 0 else ""
            return "PASS", f"ตรวจพบขวดน้ำถูกต้อง - Water bottle detected (conf: {confidence:.1f}%{label_info})"
    
    def draw_bottle_detection_overlay(self, frame, bottles, status, reason, processing_time):
        """Enhanced bottle detection overlay with confidence and detailed metrics"""
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center crosshairs with better visibility
        cv2.line(overlay_frame, (w//2-40, h//2), (w//2+40, h//2), (0, 255, 255), 3)
        cv2.line(overlay_frame, (w//2, h//2-40), (w//2, h//2+40), (0, 255, 255), 3)
        
        # Draw center tolerance circle
        cv2.circle(overlay_frame, (w//2, h//2), self.center_tolerance, (0, 255, 255), 2)
        
        # Draw detected bottles with enhanced information
        for i, bottle in enumerate(bottles):
            contour = bottle['contour']
            center = bottle['center']
            bbox = bottle['bbox']
            confidence = bottle.get('confidence', 0)
            
            # Color coding based on confidence
            if confidence >= 80:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence >= 60:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw contour with confidence-based thickness
            thickness = max(2, int(confidence / 20))
            cv2.drawContours(overlay_frame, [contour], -1, color, thickness)
            
            # Draw bounding box
            x, y, w_box, h_box = bbox
            cv2.rectangle(overlay_frame, (x, y), (x+w_box, y+h_box), color, 2)
            
            # Draw center point and centroid
            cv2.circle(overlay_frame, center, 8, (0, 0, 255), -1)
            if 'centroid' in bottle:
                cv2.circle(overlay_frame, bottle['centroid'], 6, (255, 0, 255), -1)
            
            # Draw bottle information with enhanced details
            info_y = y - 25
            label_bg_color = (0, 0, 0)
            
            # Background for text (larger for more info)
            cv2.rectangle(overlay_frame, (x-5, info_y-75), (x+250, info_y+5), label_bg_color, -1)
            
            # Bottle ranking and confidence
            cv2.putText(overlay_frame, f"WATER BOTTLE #{i+1}", 
                       (x, info_y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay_frame, f"Confidence: {confidence:.1f}%", 
                       (x, info_y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(overlay_frame, f"Area: {bottle['area']:.0f}px", 
                       (x, info_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(overlay_frame, f"Ratio: {bottle['aspect_ratio']:.2f}", 
                       (x, info_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show label confidence if available
            if 'label_confidence' in bottle and bottle['label_confidence'] > 0:
                cv2.putText(overlay_frame, f"Label: {bottle['label_confidence']:.1f}%", 
                           (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw bottle top and bottom points if available
            if 'top_point' in bottle and 'bottom_point' in bottle:
                cv2.circle(overlay_frame, bottle['top_point'], 4, (255, 255, 0), -1)
                cv2.circle(overlay_frame, bottle['bottom_point'], 4, (255, 255, 0), -1)
        
        # Enhanced status overlay with Thai text
        status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        
        # Background for status
        cv2.rectangle(overlay_frame, (5, 5), (w-5, 110), (0, 0, 0), -1)
        
        cv2.putText(overlay_frame, f"WATER BOTTLE STATUS: {status}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(overlay_frame, f"Result: {reason}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Detection parameters display for water bottles
        cv2.putText(overlay_frame, f"Min Confidence: 70% | Aspect Ratio: 2.2-3.8 | Label Detection: ON", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show bottle type optimized for
        cv2.putText(overlay_frame, f"Optimized for: Nestle Pure Life & similar water bottles", 
                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        # Enhanced stats overlay
        stats_bg_y = h - 120
        cv2.rectangle(overlay_frame, (5, stats_bg_y), (400, h-5), (0, 0, 0), -1)
        
        cv2.putText(overlay_frame, f"Bottles Found: {len(bottles)}", 
                   (10, stats_bg_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay_frame, f"Processing: {processing_time:.1f}ms", 
                   (10, stats_bg_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay_frame, f"Total Scans: {self.detection_stats['total_detections']}", 
                   (10, stats_bg_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Mode and confidence info
        mode_text = "CONTINUOUS" if self.continuous_mode else "MANUAL"
        cv2.putText(overlay_frame, f"Mode: {mode_text}", 
                   (10, stats_bg_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Controls help
        cv2.putText(overlay_frame, "SPACE:Detect S:Save C:Continuous R:Reset Q:Quit", 
                   (10, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return overlay_frame
        
    def save_bottle_detection_result(self, bottles, status, reason, processing_time):
        """Save bottle detection result to file"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'detection_type': 'bottle_detection',
            'status': status,
            'reason': reason,
            'bottles_count': len(bottles),
            'processing_time_ms': processing_time,
            'bottles': []
        }
        
        for bottle in bottles:
            result['bottles'].append({
                'area': float(bottle['area']),
                'center': bottle['center'],
                'bbox': bottle['bbox'],
                'aspect_ratio': float(bottle['aspect_ratio']),
                'circularity': float(bottle['circularity']),
                'solidity': float(bottle['solidity'])
            })
        
        # Save to JSON file
        filename = f"bottle_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Bottle detection result saved to {filename}")
        
        return filename
    def run(self):
        """Main bottle detection loop"""
        logger.info("Starting live bottle detection...")
        logger.info("Controls: SPACE=Detect, S=Save, C=Continuous, R=Reset, Q=Quit")
        logger.info(f"Water Bottle Detection Parameters (Optimized for Nestlé Pure Life):")
        logger.info(f"  - Area range: {self.min_area} - {self.max_area} pixels")
        logger.info(f"  - Aspect ratio: {self.min_aspect_ratio} - {self.max_aspect_ratio} (tall & narrow)")
        logger.info(f"  - Circularity: {self.min_circularity} - {self.max_circularity}")
        logger.info(f"  - Solidity: ≥{self.min_solidity}")
        logger.info(f"  - Extent: ≥{self.min_extent}")
        logger.info(f"  - Center tolerance: {self.center_tolerance} pixels")
        logger.info(f"  - Minimum confidence: 70% (higher for water bottles)")
        logger.info(f"  - Label detection: {'ON' if self.detect_labels else 'OFF'}")
        logger.info(f"  - Edge detection: Low={self.canny_low}, High={self.canny_high} (optimized for transparent bottles)")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                display_frame = frame.copy()
                should_detect = False
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space bar
                    should_detect = True
                elif key == ord('c'):  # Toggle continuous mode
                    self.continuous_mode = not self.continuous_mode
                    logger.info(f"Continuous bottle detection: {'ON' if self.continuous_mode else 'OFF'}")
                elif key == ord('r'):  # Reset stats
                    self.detection_stats = {
                        'total_detections': 0,
                        'objects_detected': 0,
                        'session_start': datetime.now().isoformat()
                    }
                    logger.info("Bottle detection statistics reset")
                elif key == ord('s'):  # Save frame
                    filename = f"bottle_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Bottle frame saved as {filename}")
                
                # Continuous detection mode
                if self.continuous_mode:
                    current_time = time.time()
                    if current_time - self.last_detection_time > 0.5:  # Detect every 500ms
                        should_detect = True
                        self.last_detection_time = current_time
                
                # Perform bottle detection
                if should_detect:
                    bottles, thresh, processing_time = self.detect_bottles(frame)
                    status, reason = self.analyze_bottle_detection(bottles, frame.shape)
                    
                    # Update statistics
                    self.detection_stats['total_detections'] += 1
                    self.detection_stats['objects_detected'] += len(bottles)
                    
                    # Draw overlay
                    display_frame = self.draw_bottle_detection_overlay(
                        frame, bottles, status, reason, processing_time
                    )
                    
                    # Log result
                    logger.info(f"Bottle Detection #{self.detection_stats['total_detections']}: "
                              f"{status} - {reason} ({processing_time:.1f}ms)")
                    
                    # Save result if it's a manual detection
                    if not self.continuous_mode:
                        self.save_bottle_detection_result(bottles, status, reason, processing_time)
                
                else:
                    # Show basic overlay without detection
                    h, w = frame.shape[:2]
                    cv2.putText(display_frame, "BOTTLE DETECTION READY", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Press SPACE to detect bottles", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(display_frame, "SPACE:Detect S:Save C:Continuous R:Reset Q:Quit", 
                               (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Display frame
                cv2.imshow('Live Bottle Detection System', display_frame)
                
        except KeyboardInterrupt:
            logger.info("Bottle detection stopped by user")
        except Exception as e:
            logger.error(f"Error during bottle detection: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        logger.info("=" * 50)
        logger.info("BOTTLE DETECTION SESSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total bottle scans: {self.detection_stats['total_detections']}")
        logger.info(f"Bottles detected: {self.detection_stats['objects_detected']}")
        if self.detection_stats['total_detections'] > 0:
            avg_bottles = self.detection_stats['objects_detected'] / self.detection_stats['total_detections']
            logger.info(f"Average bottles per scan: {avg_bottles:.2f}")

def main():
    """Main function"""
    print("Water Bottle Detection System")
    print("=" * 60)
    print("ระบบตรวจจับขวดน้ำ (เหมาะสำหรับ Nestlé Pure Life)")
    print("=" * 60)
    print("Optimized Features:")
    print("- Transparent bottle detection with edge enhancement")
    print("- Blue label detection (HSV color analysis)")
    print("- Neck narrowing detection for water bottles")
    print("- Aspect ratio tuned for tall, narrow bottles (2.2-3.8)")
    print("- Higher confidence threshold (70%) for accuracy")
    print("- Label confidence scoring for branded bottles")
    print("=" * 60)
    print("Detection Criteria for Water Bottles:")
    print("- Minimum confidence: 70%")
    print("- Area: 4000-45000 pixels (medium-large)")
    print("- Aspect ratio: 2.2-3.8 (tall & narrow)")
    print("- Circularity: 0.20-0.70 (bottle-like shape)")
    print("- Solidity: ≥75% (mostly solid)")
    print("- Extent: ≥65% (fills bounding box)")
    print("- Label detection: Blue/Cyan HSV range")
    print("- Neck narrowing: Detected for water bottle validation")
    print("=" * 60)
    
    # Try to find available camera
    camera_id = 0
    detector = None
    
    try:
        detector = LiveCameraDetector(camera_id)
        detector.run()
    except RuntimeError as e:
        print(f"Camera Error: {e}")
        print("\nTroubleshooting:")
        print("1. ตรวจสอบการเชื่อมต่อกล้อง - Check camera connection")
        print("2. ลองเปลี่ยน camera_id (0, 1, 2...) - Try different camera_id")
        print("3. ปิดแอปอื่นที่ใช้กล้อง - Close other apps using camera")
        print("4. ตรวจสอบสิทธิ์กล้องใน Linux - Check camera permissions on Linux")
        print("\nFor optimal water bottle detection:")
        print("- ใช้แสงที่เพียงพอและสม่ำเสมอ - Use adequate, even lighting")
        print("- วางขวดตั้งตรง - Position bottle upright")
        print("- ให้ขวดมองเห็นชัดเจน - Ensure bottle is clearly visible")
        print("- หลีกเลี่ยงพื้นหลังที่รกรุงรัง - Use clean background")
        print("- วางขวดไว้กึ่งกลางเฟรม - Center bottle in frame")
        print("- ฉลากขวดช่วยเพิ่มความแม่นยำ - Labels help improve accuracy")
        print("- ขวดโปร่งใสทำงานได้ดีที่สุด - Transparent bottles work best")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if detector:
            detector.cleanup()

if __name__ == "__main__":
    main()