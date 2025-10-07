#!/usr/bin/env python3
"""
Headless Vision System Demo
Demonstrates computer vision processing without GUI display
"""

import cv2
import numpy as np
import time
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeadlessVisionDemo:
    """Vision system demonstration for headless environments"""
    
    def __init__(self):
        self.min_area = 1000
        self.max_area = 50000
        self.test_images = self.create_test_images()
        self.results = []
        
        logger.info("Headless vision demo initialized")
    
    def create_test_images(self):
        """Create synthetic test images"""
        images = []
        
        # Test scenarios
        scenarios = [
            ("Good Part - Centered", lambda img: cv2.circle(img, (320, 240), 80, (200, 200, 200), -1)),
            ("Bad Part - Off Center", lambda img: cv2.circle(img, (400, 150), 70, (200, 200, 200), -1)),
            ("Multiple Objects", lambda img: [cv2.circle(img, (200, 200), 50, (200, 200, 200), -1), 
                                            cv2.circle(img, (400, 300), 60, (200, 200, 200), -1)]),
            ("No Objects", lambda img: None),
            ("Object Too Small", lambda img: cv2.circle(img, (320, 240), 20, (200, 200, 200), -1)),
            ("Perfect Part with Noise", lambda img: [cv2.circle(img, (320, 240), 75, (200, 200, 200), -1),
                                                    cv2.add(img, np.random.randint(0, 30, img.shape, dtype=np.uint8))])
        ]
        
        for name, draw_func in scenarios:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img.fill(50)  # Dark background
            
            try:
                result = draw_func(img)
                if isinstance(result, list):
                    pass  # Multiple operations already applied
            except:
                pass  # No objects or error
            
            images.append((name, img))
        
        return images
    
    def detect_objects(self, image):
        """Object detection using contours"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                detected_objects.append({
                    'area': area,
                    'center': (center_x, center_y),
                    'bounding_box': (x, y, w, h)
                })
        
        return detected_objects
    
    def inspect_part(self, image):
        """Perform quality inspection"""
        objects = self.detect_objects(image)
        
        if len(objects) == 1:
            obj = objects[0]
            area_ok = 5000 <= obj['area'] <= 25000
            
            center_x, center_y = obj['center']
            image_center_x, image_center_y = image.shape[1] // 2, image.shape[0] // 2
            position_ok = (abs(center_x - image_center_x) < 50 and 
                          abs(center_y - image_center_y) < 50)
            
            pass_inspection = area_ok and position_ok
            
            result = {
                'pass': pass_inspection,
                'object_count': len(objects),
                'area': float(obj['area']),
                'position': obj['center'],
                'area_ok': area_ok,
                'position_ok': position_ok,
                'timestamp': time.time()
            }
        else:
            result = {
                'pass': False,
                'object_count': len(objects),
                'area': 0.0,
                'position': (0, 0),
                'area_ok': False,
                'position_ok': False,
                'timestamp': time.time()
            }
        
        return result, objects
    
    def run_batch_inspection(self):
        """Run batch inspection on all test images"""
        logger.info("Starting batch inspection...")
        
        total_inspections = 0
        passed_inspections = 0
        
        for image_name, image in self.test_images:
            logger.info(f"Processing: {image_name}")
            
            start_time = time.time()
            result, objects = self.inspect_part(image)
            processing_time = time.time() - start_time
            
            result['image_name'] = image_name
            result['processing_time'] = processing_time
            
            total_inspections += 1
            if result['pass']:
                passed_inspections += 1
            
            # Log detailed results
            status = "PASS" if result['pass'] else "FAIL"
            logger.info(f"  Result: {status}")
            logger.info(f"  Objects detected: {result['object_count']}")
            logger.info(f"  Area: {result['area']:.1f} pixels")
            logger.info(f"  Position: {result['position']}")
            logger.info(f"  Processing time: {processing_time*1000:.1f}ms")
            
            self.results.append(result)
        
        # Calculate statistics
        pass_rate = (passed_inspections / total_inspections) * 100 if total_inspections > 0 else 0
        avg_processing_time = np.mean([r['processing_time'] for r in self.results]) * 1000
        
        logger.info("=" * 50)
        logger.info("INSPECTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total inspections: {total_inspections}")
        logger.info(f"Passed: {passed_inspections}")
        logger.info(f"Failed: {total_inspections - passed_inspections}")
        logger.info(f"Pass rate: {pass_rate:.1f}%")
        logger.info(f"Average processing time: {avg_processing_time:.1f}ms")
        
        return self.results
    
    def save_results(self, filename="inspection_results.json"):
        """Save inspection results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def generate_report(self):
        """Generate detailed inspection report"""
        if not self.results:
            logger.warning("No results available for report")
            return
        
        report = []
        report.append("VISION SYSTEM INSPECTION REPORT")
        report.append("=" * 40)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        total = len(self.results)
        passed = sum(1 for r in self.results if r['pass'])
        failed = total - passed
        
        report.append("SUMMARY:")
        report.append(f"  Total Inspections: {total}")
        report.append(f"  Passed: {passed}")
        report.append(f"  Failed: {failed}")
        report.append(f"  Pass Rate: {(passed/total)*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for i, result in enumerate(self.results, 1):
            status = "PASS" if result['pass'] else "FAIL"
            report.append(f"{i:2d}. {result['image_name']}")
            report.append(f"    Status: {status}")
            report.append(f"    Objects: {result['object_count']}")
            report.append(f"    Area: {result['area']:.1f}")
            report.append(f"    Position: {result['position']}")
            report.append(f"    Time: {result['processing_time']*1000:.1f}ms")
            report.append("")
        
        # Performance metrics
        processing_times = [r['processing_time'] * 1000 for r in self.results]
        report.append("PERFORMANCE METRICS:")
        report.append(f"  Average processing time: {np.mean(processing_times):.1f}ms")
        report.append(f"  Min processing time: {np.min(processing_times):.1f}ms")
        report.append(f"  Max processing time: {np.max(processing_times):.1f}ms")
        report.append(f"  Estimated throughput: {1000/np.mean(processing_times):.1f} parts/second")
        
        # Save report
        report_text = "\n".join(report)
        try:
            with open("inspection_report.txt", "w") as f:
                f.write(report_text)
            logger.info("Report saved to inspection_report.txt")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
        
        # Print report to console
        print("\n" + report_text)

def main():
    """Main demonstration function"""
    print("Headless Vision System Demo")
    print("=" * 40)
    
    try:
        demo = HeadlessVisionDemo()
        
        # Run batch inspection
        results = demo.run_batch_inspection()
        
        # Save results
        demo.save_results()
        
        # Generate report
        demo.generate_report()
        
        print("\nDemo completed successfully!")
        print("Check 'inspection_results.json' and 'inspection_report.txt' for detailed results.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()