#!/usr/bin/env python3
"""
Vision System Simulation Demo
Demonstrates basic computer vision operations without requiring camera hardware
"""

import cv2
import numpy as np
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionSimulationDemo:
    """Vision system demonstration using simulated images"""
    
    def __init__(self):
        # Vision parameters
        self.min_area = 1000  # Minimum object area
        self.max_area = 50000  # Maximum object area
        
        # Create some test images
        self.test_images = self.create_test_images()
        self.current_image_index = 0
        
        logger.info("Vision simulation demo initialized")
    
    def create_test_images(self):
        """Create synthetic test images for demonstration"""
        images = []
        
        # Test image 1: Good part (centered circle)
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img1.fill(50)  # Dark background
        cv2.circle(img1, (320, 240), 80, (200, 200, 200), -1)  # Good part
        images.append(("Good Part - Centered", img1))
        
        # Test image 2: Bad part (off-center)
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2.fill(50)
        cv2.circle(img2, (400, 150), 70, (200, 200, 200), -1)  # Off-center part
        images.append(("Bad Part - Off Center", img2))
        
        # Test image 3: Multiple objects
        img3 = np.zeros((480, 640, 3), dtype=np.uint8)
        img3.fill(50)
        cv2.circle(img3, (200, 200), 50, (200, 200, 200), -1)
        cv2.circle(img3, (400, 300), 60, (200, 200, 200), -1)
        images.append(("Multiple Objects", img3))
        
        # Test image 4: No objects
        img4 = np.zeros((480, 640, 3), dtype=np.uint8)
        img4.fill(50)
        images.append(("No Objects", img4))
        
        # Test image 5: Too small object
        img5 = np.zeros((480, 640, 3), dtype=np.uint8)
        img5.fill(50)
        cv2.circle(img5, (320, 240), 20, (200, 200, 200), -1)  # Too small
        images.append(("Object Too Small", img5))
        
        # Test image 6: Perfect part with noise
        img6 = np.zeros((480, 640, 3), dtype=np.uint8)
        img6.fill(50)
        cv2.circle(img6, (320, 240), 75, (200, 200, 200), -1)
        # Add some noise
        noise = np.random.randint(0, 30, img6.shape, dtype=np.uint8)
        img6 = cv2.add(img6, noise)
        images.append(("Good Part with Noise", img6))
        
        return images
    
    def get_current_image(self):
        """Get current test image"""
        if not self.test_images:
            return None, "No Image"
        
        name, image = self.test_images[self.current_image_index]
        return image, name
    
    def next_image(self):
        """Move to next test image"""
        if self.test_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.test_images)
    
    def previous_image(self):
        """Move to previous test image"""
        if self.test_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.test_images)
    
    def detect_objects(self, image):
        """Simple object detection using contours"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to create binary image
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        
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
    
    def draw_results(self, image, result, objects, image_name):
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
        
        # Draw image name
        cv2.putText(result_image, image_name, 
                   (10, result_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw criteria
        criteria_y = 90
        cv2.putText(result_image, f"Area OK: {result['area_ok']}", 
                   (10, criteria_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_image, f"Position OK: {result['position_ok']}", 
                   (10, criteria_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def run_demo(self):
        """Run interactive vision demonstration"""
        logger.info("Starting vision simulation demo")
        logger.info("Controls:")
        logger.info("  SPACE - Next image")
        logger.info("  B     - Previous image")
        logger.info("  S     - Save current result")
        logger.info("  Q     - Quit")
        
        inspection_count = 0
        pass_count = 0
        
        try:
            while True:
                # Get current image
                image, image_name = self.get_current_image()
                if image is None:
                    break
                
                # Perform inspection
                result, objects = self.inspect_part(image)
                inspection_count += 1
                
                if result['pass']:
                    pass_count += 1
                
                # Draw results
                display_image = self.draw_results(image, result, objects, image_name)
                
                # Add statistics
                pass_rate = (pass_count / inspection_count) * 100 if inspection_count > 0 else 0
                cv2.putText(display_image, f"Pass Rate: {pass_rate:.1f}%", 
                           (10, display_image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, f"Total: {inspection_count}", 
                           (10, display_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display image
                cv2.imshow('Vision System Simulation', display_image)
                
                # Log result
                logger.info(f"Image: {image_name} | Result: {'PASS' if result['pass'] else 'FAIL'} | "
                           f"Objects: {result['object_count']} | Area: {result['area']}")
                
                # Handle keyboard input
                key = cv2.waitKey(0) & 0xFF  # Wait for key press
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord(' '):  # Space - next image
                    self.next_image()
                elif key == ord('b') or key == ord('B'):  # B - previous image
                    self.previous_image()
                elif key == ord('s') or key == ord('S'):  # S - save image
                    timestamp = int(time.time())
                    filename = f"inspection_result_{timestamp}.jpg"
                    cv2.imwrite(filename, display_image)
                    logger.info(f"Result saved as {filename}")
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            
            # Final statistics
            logger.info(f"Demo complete: {inspection_count} inspections, {pass_count} passed")
            logger.info(f"Overall pass rate: {(pass_count/inspection_count)*100:.1f}%" if inspection_count > 0 else "No inspections")

def test_opencv_installation():
    """Test if OpenCV is properly installed"""
    try:
        print(f"OpenCV version: {cv2.__version__}")
        
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_img, (50, 50), 30, (0, 255, 0), -1)
        
        print("OpenCV is working correctly!")
        return True
        
    except Exception as e:
        print(f"OpenCV test failed: {e}")
        return False

def main():
    """Main demonstration function"""
    print("Vision System Simulation Demo")
    print("=" * 40)
    
    # Test OpenCV
    if not test_opencv_installation():
        print("Please install OpenCV: pip install opencv-python")
        return
    
    # Run demo
    demo = VisionSimulationDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()