#!/usr/bin/env python3
"""
Simple Vision System Demo
Demonstrates basic computer vision operations for assembly line inspection
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
from gpiozero import LED, Button
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVisionDemo:
    """Basic vision system demonstration"""
    
    def __init__(self):
        # Initialize camera
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_video_configuration(
            main={"format": 'RGB888', "size": (640, 480)}
        ))
        
        # Initialize GPIO (if available)
        try:
            self.trigger_button = Button(2)
            self.pass_led = LED(18)
            self.fail_led = LED(19)
            self.gpio_available = True
            logger.info("GPIO initialized successfully")
        except Exception as e:
            logger.warning(f"GPIO not available: {e}")
            self.gpio_available = False
        
        # Vision parameters
        self.min_area = 1000  # Minimum object area
        self.max_area = 50000  # Maximum object area
        
        logger.info("Vision demo initialized")
    
    def start_camera(self):
        """Start camera capture"""
        self.camera.start()
        time.sleep(2)  # Allow camera to warm up
        logger.info("Camera started")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera.stop()
        logger.info("Camera stopped")
    
    def capture_image(self):
        """Capture single image from camera"""
        try:
            frame = self.camera.capture_array()
            return frame
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return None
    
    def detect_objects(self, image):
        """Simple object detection using contours"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to create binary image
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area <= area <= self.max_area:
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                detected_objects.append({
                    'area': area,
                    'center': (center_x, center_y),
                    'bounding_box': (x, y, w, h),
                    'contour': contour
                })
        
        return detected_objects
    
    def inspect_part(self, image):
        """Perform part inspection"""
        objects = self.detect_objects(image)
        
        # Simple pass/fail logic
        if len(objects) == 1:
            obj = objects[0]
            # Check if object size is within acceptable range
            area_ok = 5000 <= obj['area'] <= 25000
            
            # Check if object is centered (simple position check)
            center_x, center_y = obj['center']
            image_center_x, image_center_y = image.shape[1] // 2, image.shape[0] // 2
            position_ok = (abs(center_x - image_center_x) < 50 and 
                          abs(center_y - image_center_y) < 50)
            
            pass_inspection = area_ok and position_ok
            
            result = {
                'pass': pass_inspection,
                'object_count': len(objects),
                'area': obj['area'],
                'position': obj['center'],
                'area_ok': area_ok,
                'position_ok': position_ok
            }
        else:
            # Wrong number of objects
            result = {
                'pass': False,
                'object_count': len(objects),
                'area': 0,
                'position': (0, 0),
                'area_ok': False,
                'position_ok': False
            }
        
        return result, objects
    
    def draw_results(self, image, result, objects):
        """Draw inspection results on image"""
        result_image = image.copy()
        
        # Draw detected objects
        for obj in objects:
            x, y, w, h = obj['bounding_box']
            center = obj['center']
            
            # Draw bounding rectangle
            color = (0, 255, 0) if result['pass'] else (0, 0, 255)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(result_image, center, 5, color, -1)
            
            # Draw area text
            cv2.putText(result_image, f"Area: {obj['area']}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw overall result
        result_text = "PASS" if result['pass'] else "FAIL"
        text_color = (0, 255, 0) if result['pass'] else (0, 0, 255)
        cv2.putText(result_image, result_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Draw object count
        cv2.putText(result_image, f"Objects: {result['object_count']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def update_outputs(self, result):
        """Update LED outputs based on inspection result"""
        if not self.gpio_available:
            return
        
        try:
            if result['pass']:
                self.pass_led.on()
                self.fail_led.off()
            else:
                self.pass_led.off()
                self.fail_led.on()
        except Exception as e:
            logger.error(f"Failed to update outputs: {e}")
    
    def run_continuous_demo(self):
        """Run continuous vision demonstration"""
        logger.info("Starting continuous vision demo")
        logger.info("Press 'q' to quit, 's' to save image, SPACE for single inspection")
        
        self.start_camera()
        
        try:
            while True:
                # Capture frame
                frame = self.capture_image()
                if frame is None:
                    continue
                
                # Perform inspection
                result, objects = self.inspect_part(frame)
                
                # Draw results
                display_image = self.draw_results(frame, result, objects)
                
                # Update hardware outputs
                self.update_outputs(result)
                
                # Display image
                cv2.imshow('Vision System Demo', display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current image
                    timestamp = int(time.time())
                    filename = f"vision_demo_{timestamp}.jpg"
                    cv2.imwrite(filename, cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
                    logger.info(f"Image saved as {filename}")
                elif key == ord(' '):
                    # Print detailed inspection result
                    logger.info(f"Inspection Result: {result}")
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()
            
            # Turn off LEDs
            if self.gpio_available:
                self.pass_led.off()
                self.fail_led.off()
    
    def run_triggered_demo(self):
        """Run triggered vision demonstration (requires button)"""
        if not self.gpio_available:
            logger.error("GPIO not available for triggered demo")
            return
        
        logger.info("Starting triggered vision demo")
        logger.info("Press button to trigger inspection")
        
        self.start_camera()
        
        def trigger_inspection():
            logger.info("Inspection triggered!")
            
            # Capture image
            frame = self.capture_image()
            if frame is None:
                return
            
            # Perform inspection
            result, objects = self.inspect_part(frame)
            
            # Update outputs
            self.update_outputs(result)
            
            # Log result
            logger.info(f"Inspection complete: {result}")
            
            # Save image with timestamp
            timestamp = int(time.time())
            filename = f"triggered_inspection_{timestamp}.jpg"
            result_image = self.draw_results(frame, result, objects)
            cv2.imwrite(filename, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Result saved as {filename}")
        
        # Setup button callback
        self.trigger_button.when_pressed = trigger_inspection
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            self.stop_camera()
            if self.gpio_available:
                self.pass_led.off()
                self.fail_led.off()

def main():
    """Main demonstration function"""
    demo = SimpleVisionDemo()
    
    print("Vision System Demo")
    print("1. Continuous mode")
    print("2. Triggered mode")
    
    try:
        choice = input("Select mode (1 or 2): ").strip()
        
        if choice == "1":
            demo.run_continuous_demo()
        elif choice == "2":
            demo.run_triggered_demo()
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nDemo terminated")

if __name__ == "__main__":
    main()