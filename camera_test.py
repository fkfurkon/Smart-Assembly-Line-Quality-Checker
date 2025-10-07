#!/usr/bin/env python3
"""
Camera Test and Calibration
Test camera functionality and perform basic calibration
"""

import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraTest:
    """Camera testing and calibration utilities"""
    
    def __init__(self):
        self.camera = None
        self.calibration_data = None
        
        # Test patterns
        self.test_images = []
        
    def initialize_camera(self, resolution=(1920, 1080)):
        """Initialize camera with specified resolution"""
        try:
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_video_configuration(
                main={"format": 'RGB888', "size": resolution}
            )
            self.camera.configure(config)
            
            logger.info(f"Camera initialized with resolution {resolution}")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def start_camera(self):
        """Start camera capture"""
        if self.camera:
            self.camera.start()
            time.sleep(2)  # Allow camera to stabilize
            logger.info("Camera started")
            return True
        return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
    
    def capture_test_image(self, filename=None):
        """Capture test image"""
        if not self.camera:
            logger.error("Camera not initialized")
            return None
        
        try:
            frame = self.camera.capture_array()
            
            if filename:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, bgr_frame)
                logger.info(f"Test image saved as {filename}")
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return None
    
    def test_camera_modes(self):
        """Test different camera modes and settings"""
        logger.info("Testing different camera modes...")
        
        # Test different resolutions
        resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080)
        ]
        
        for resolution in resolutions:
            logger.info(f"Testing resolution: {resolution}")
            
            if self.camera:
                self.camera.stop()
            
            if self.initialize_camera(resolution):
                self.start_camera()
                
                # Capture test image
                filename = f"test_{resolution[0]}x{resolution[1]}.jpg"
                self.capture_test_image(filename)
                
                # Test image properties
                frame = self.capture_test_image()
                if frame is not None:
                    logger.info(f"Image shape: {frame.shape}")
                    logger.info(f"Image dtype: {frame.dtype}")
                    logger.info(f"Image size: {frame.size} pixels")
                
                time.sleep(1)
        
        logger.info("Camera mode testing complete")
    
    def test_exposure_settings(self):
        """Test different exposure settings"""
        logger.info("Testing exposure settings...")
        
        if not self.camera:
            self.initialize_camera()
            self.start_camera()
        
        # Test different exposure times (microseconds)
        exposure_times = [1000, 5000, 10000, 20000, 50000]
        
        for exposure_time in exposure_times:
            try:
                # Set exposure time
                self.camera.set_controls({"ExposureTime": exposure_time})
                time.sleep(1)  # Allow setting to take effect
                
                # Capture image
                filename = f"exposure_{exposure_time}us.jpg"
                frame = self.capture_test_image(filename)
                
                if frame is not None:
                    # Calculate image brightness
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    brightness = np.mean(gray)
                    logger.info(f"Exposure {exposure_time}μs: Brightness = {brightness:.1f}")
                
            except Exception as e:
                logger.error(f"Failed to set exposure {exposure_time}: {e}")
        
        logger.info("Exposure testing complete")
    
    def perform_camera_calibration(self, pattern_size=(9, 6), square_size=25.0):
        """Perform camera calibration using chessboard pattern"""
        logger.info("Starting camera calibration...")
        logger.info("Present chessboard pattern to camera and press 's' to capture")
        logger.info("Collect at least 10 images, then press 'c' to calibrate")
        logger.info("Press 'q' to quit")
        
        if not self.camera:
            self.initialize_camera()
            self.start_camera()
        
        # Calibration parameters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Scale by actual square size in mm
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        capture_count = 0
        
        try:
            while True:
                # Capture frame
                frame = self.capture_test_image()
                if frame is None:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
                
                # Draw the frame
                display_frame = frame.copy()
                
                if ret:
                    # Refine corner positions
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Draw corners
                    cv2.drawChessboardCorners(display_frame, pattern_size, corners2, ret)
                    
                    # Add text
                    cv2.putText(display_frame, "Pattern found - Press 's' to capture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No pattern found", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show capture count
                cv2.putText(display_frame, f"Captured: {capture_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display image
                cv2.imshow('Camera Calibration', cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s') and ret:
                    # Save calibration image
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    capture_count += 1
                    
                    # Save image
                    filename = f"calibration_{capture_count:02d}.jpg"
                    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    logger.info(f"Calibration image {capture_count} captured")
                
                elif key == ord('c') and capture_count >= 4:
                    # Perform calibration
                    logger.info("Performing camera calibration...")
                    
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                        objpoints, imgpoints, gray.shape[::-1], None, None
                    )
                    
                    if ret:
                        self.calibration_data = {
                            'camera_matrix': mtx,
                            'distortion_coefficients': dist,
                            'rotation_vectors': rvecs,
                            'translation_vectors': tvecs,
                            'image_size': gray.shape[::-1]
                        }
                        
                        # Save calibration data
                        np.savez('camera_calibration.npz',
                                camera_matrix=mtx,
                                distortion_coefficients=dist)
                        
                        logger.info("Camera calibration completed and saved")
                        logger.info(f"Camera matrix:\n{mtx}")
                        logger.info(f"Distortion coefficients: {dist.ravel()}")
                        
                        # Calculate reprojection error
                        mean_error = 0
                        for i in range(len(objpoints)):
                            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                            mean_error += error
                        
                        mean_error /= len(objpoints)
                        logger.info(f"Mean reprojection error: {mean_error}")
                    
                    else:
                        logger.error("Calibration failed")
        
        except KeyboardInterrupt:
            logger.info("Calibration interrupted")
        
        finally:
            cv2.destroyAllWindows()
    
    def test_undistortion(self):
        """Test image undistortion using calibration data"""
        if self.calibration_data is None:
            # Try to load calibration data
            try:
                data = np.load('camera_calibration.npz')
                self.calibration_data = {
                    'camera_matrix': data['camera_matrix'],
                    'distortion_coefficients': data['distortion_coefficients']
                }
                logger.info("Calibration data loaded")
            except:
                logger.error("No calibration data available")
                return
        
        if not self.camera:
            self.initialize_camera()
            self.start_camera()
        
        logger.info("Testing image undistortion - Press 'q' to quit")
        
        try:
            while True:
                # Capture frame
                frame = self.capture_test_image()
                if frame is None:
                    continue
                
                # Convert to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Undistort image
                mtx = self.calibration_data['camera_matrix']
                dist = self.calibration_data['distortion_coefficients']
                
                undistorted = cv2.undistort(bgr_frame, mtx, dist, None, mtx)
                
                # Show original and undistorted side by side
                comparison = np.hstack((bgr_frame, undistorted))
                
                # Add labels
                cv2.putText(comparison, "Original", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(comparison, "Undistorted", (bgr_frame.shape[1] + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Undistortion Test', comparison)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("Undistortion test interrupted")
        
        finally:
            cv2.destroyAllWindows()
    
    def test_focus_measurement(self):
        """Test focus measurement using variance of Laplacian"""
        logger.info("Testing focus measurement - Move object to test focus")
        logger.info("Press 'q' to quit")
        
        if not self.camera:
            self.initialize_camera()
            self.start_camera()
        
        try:
            while True:
                # Capture frame
                frame = self.capture_test_image()
                if frame is None:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Calculate focus measure (variance of Laplacian)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                focus_measure = laplacian.var()
                
                # Display frame with focus measure
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(display_frame, f"Focus: {focus_measure:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Color code focus quality
                if focus_measure > 500:
                    color = (0, 255, 0)  # Good focus
                    quality = "GOOD"
                elif focus_measure > 200:
                    color = (0, 255, 255)  # Moderate focus
                    quality = "MODERATE"
                else:
                    color = (0, 0, 255)  # Poor focus
                    quality = "POOR"
                
                cv2.putText(display_frame, quality, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv2.imshow('Focus Test', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("Focus test interrupted")
        
        finally:
            cv2.destroyAllWindows()

def main():
    """Main test function"""
    camera_test = CameraTest()
    
    print("Camera Test and Calibration")
    print("1. Test camera modes")
    print("2. Test exposure settings")
    print("3. Perform camera calibration")
    print("4. Test undistortion")
    print("5. Test focus measurement")
    
    try:
        choice = input("Select test (1-5): ").strip()
        
        if choice == "1":
            camera_test.test_camera_modes()
        elif choice == "2":
            camera_test.test_exposure_settings()
        elif choice == "3":
            camera_test.perform_camera_calibration()
        elif choice == "4":
            camera_test.test_undistortion()
        elif choice == "5":
            camera_test.test_focus_measurement()
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nTest terminated")
    
    finally:
        camera_test.stop_camera()

if __name__ == "__main__":
    main()