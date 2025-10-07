#!/usr/bin/env python3
"""
GPIO Control Demo
Demonstrates basic GPIO operations for industrial I/O
"""

import time
import threading
from gpiozero import LED, Button, PWMOutputDevice
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPIODemo:
    """GPIO demonstration for industrial applications"""
    
    def __init__(self):
        # Initialize outputs
        try:
            self.status_led = LED(18)      # Status indicator
            self.alarm_led = LED(19)       # Alarm indicator
            self.process_led = LED(20)     # Process running indicator
            self.strobe_light = PWMOutputDevice(21)  # Strobe light with PWM
            
            # Initialize inputs
            self.start_button = Button(2, pull_up=True, bounce_time=0.1)
            self.stop_button = Button(3, pull_up=True, bounce_time=0.1)
            self.sensor_input = Button(4, pull_up=True, bounce_time=0.05)
            
            self.gpio_available = True
            logger.info("GPIO initialized successfully")
            
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")
            self.gpio_available = False
        
        # Demo state
        self.running = False
        self.alarm_active = False
        self.sensor_count = 0
        
        # Setup button callbacks
        if self.gpio_available:
            self.start_button.when_pressed = self.start_process
            self.stop_button.when_pressed = self.stop_process
            self.sensor_input.when_pressed = self.sensor_triggered
    
    def start_process(self):
        """Start process when start button is pressed"""
        if not self.running:
            logger.info("Process started by button")
            self.running = True
            self.status_led.on()
            self.process_led.on()
            self.alarm_active = False
            self.alarm_led.off()
    
    def stop_process(self):
        """Stop process when stop button is pressed"""
        if self.running:
            logger.info("Process stopped by button")
            self.running = False
            self.status_led.off()
            self.process_led.off()
            self.strobe_light.off()
    
    def sensor_triggered(self):
        """Handle sensor trigger"""
        if self.running:
            self.sensor_count += 1
            logger.info(f"Sensor triggered - Count: {self.sensor_count}")
            
            # Flash strobe light
            self.strobe_light.pulse(fade_in_time=0.1, fade_out_time=0.1, n=1)
            
            # Check for alarm condition (example: too many triggers)
            if self.sensor_count > 10:
                self.trigger_alarm("Too many sensor triggers")
    
    def trigger_alarm(self, reason):
        """Trigger alarm condition"""
        logger.warning(f"ALARM: {reason}")
        self.alarm_active = True
        self.alarm_led.on()
        
        # Flash alarm LED
        def flash_alarm():
            for _ in range(10):
                if not self.alarm_active:
                    break
                self.alarm_led.on()
                time.sleep(0.5)
                self.alarm_led.off()
                time.sleep(0.5)
        
        alarm_thread = threading.Thread(target=flash_alarm)
        alarm_thread.start()
    
    def reset_alarm(self):
        """Reset alarm condition"""
        logger.info("Alarm reset")
        self.alarm_active = False
        self.alarm_led.off()
        self.sensor_count = 0
    
    def run_output_test(self):
        """Test all outputs sequentially"""
        logger.info("Starting output test sequence")
        
        if not self.gpio_available:
            logger.error("GPIO not available")
            return
        
        outputs = [
            ("Status LED", self.status_led),
            ("Alarm LED", self.alarm_led),
            ("Process LED", self.process_led)
        ]
        
        # Test each LED
        for name, led in outputs:
            logger.info(f"Testing {name}")
            led.on()
            time.sleep(1)
            led.off()
            time.sleep(0.5)
        
        # Test PWM strobe
        logger.info("Testing Strobe Light (PWM)")
        for intensity in [0.25, 0.5, 0.75, 1.0]:
            logger.info(f"Strobe intensity: {intensity*100}%")
            self.strobe_light.value = intensity
            time.sleep(1)
        
        self.strobe_light.off()
        
        # Test pulse effect
        logger.info("Testing pulse effects")
        self.status_led.pulse(fade_in_time=0.5, fade_out_time=0.5, n=3)
        time.sleep(4)
        
        logger.info("Output test complete")
    
    def run_input_test(self):
        """Test input monitoring"""
        logger.info("Starting input test - press buttons and trigger sensor")
        logger.info("Press Ctrl+C to stop test")
        
        if not self.gpio_available:
            logger.error("GPIO not available")
            return
        
        start_time = time.time()
        
        try:
            while True:
                # Print status every 5 seconds
                if time.time() - start_time > 5:
                    logger.info(f"Running: {self.running}, Alarm: {self.alarm_active}, "
                              f"Sensor count: {self.sensor_count}")
                    start_time = time.time()
                
                # Update process LED blinking when running
                if self.running:
                    self.process_led.blink(on_time=0.5, off_time=0.5)
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("Input test stopped")
        
        finally:
            # Clean up
            self.stop_process()
            self.reset_alarm()
    
    def run_automatic_sequence(self):
        """Run automatic demonstration sequence"""
        logger.info("Starting automatic demonstration sequence")
        
        if not self.gpio_available:
            logger.error("GPIO not available")
            return
        
        try:
            # Startup sequence
            logger.info("System startup...")
            self.status_led.pulse(fade_in_time=0.5, fade_out_time=0.5, n=3)
            time.sleep(3)
            
            # Start process
            self.start_process()
            time.sleep(2)
            
            # Simulate sensor triggers
            logger.info("Simulating sensor triggers...")
            for i in range(5):
                logger.info(f"Simulated trigger {i+1}")
                self.sensor_triggered()
                time.sleep(1.5)
            
            # Simulate alarm condition
            logger.info("Simulating alarm condition...")
            self.trigger_alarm("Demonstration alarm")
            time.sleep(5)
            
            # Reset alarm
            self.reset_alarm()
            time.sleep(2)
            
            # Stop process
            self.stop_process()
            
            # Shutdown sequence
            logger.info("System shutdown...")
            for led in [self.status_led, self.alarm_led, self.process_led]:
                led.on()
                time.sleep(0.3)
                led.off()
            
            logger.info("Automatic sequence complete")
        
        except KeyboardInterrupt:
            logger.info("Automatic sequence interrupted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.gpio_available:
            logger.info("Cleaning up GPIO")
            self.status_led.off()
            self.alarm_led.off()
            self.process_led.off()
            self.strobe_light.off()

def main():
    """Main demonstration function"""
    demo = GPIODemo()
    
    if not demo.gpio_available:
        print("GPIO not available - check hardware setup")
        return
    
    print("GPIO Control Demo")
    print("1. Output test")
    print("2. Input test (interactive)")
    print("3. Automatic sequence")
    
    try:
        choice = input("Select test (1, 2, or 3): ").strip()
        
        if choice == "1":
            demo.run_output_test()
        elif choice == "2":
            demo.run_input_test()
        elif choice == "3":
            demo.run_automatic_sequence()
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nDemo terminated")
    
    finally:
        demo.cleanup()

if __name__ == "__main__":
    main()