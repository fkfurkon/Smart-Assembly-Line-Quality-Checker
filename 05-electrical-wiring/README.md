# Basic Electrical Wiring for Vision Systems

## Overview
This section covers the essential electrical wiring knowledge needed to integrate Raspberry Pi vision systems into industrial environments. We'll cover power supplies, sensor interfaces, actuator control, and safety considerations for assembly line applications.

## Table of Contents
1. [Electrical Safety and Standards](#electrical-safety-and-standards)
2. [Power Supply Design](#power-supply-design)
3. [GPIO and Digital I/O](#gpio-and-digital-io)
4. [Sensor Integration](#sensor-integration)
5. [Actuator Control](#actuator-control)
6. [Industrial Communication](#industrial-communication)
7. [Grounding and Shielding](#grounding-and-shielding)
8. [Troubleshooting](#troubleshooting)

## Electrical Safety and Standards

### Industrial Safety Standards
- **IEC 61131**: Programmable controller standard
- **IEC 60204**: Safety of machinery - Electrical equipment
- **NEMA Standards**: Enclosure and component ratings
- **UL Standards**: Safety certification requirements
- **CE Marking**: European conformity standards

### Safety Practices
1. **Lock-Out Tag-Out (LOTO)**: Always de-energize before work
2. **Personal Protective Equipment (PPE)**: Safety glasses, gloves
3. **Proper Tool Usage**: Insulated tools for electrical work
4. **Voltage Testing**: Always verify circuits are de-energized
5. **Emergency Procedures**: Know location of emergency stops

### Voltage Classifications
```
Extra Low Voltage (ELV):     < 50V AC, < 120V DC
Low Voltage (LV):           50-1000V AC, 120-1500V DC
Medium Voltage (MV):        1-35 kV
High Voltage (HV):          > 35 kV
```

### Common Industrial Voltages
- **Control Circuits**: 24V DC, 120V AC
- **Motor Power**: 240V, 480V, 600V (3-phase)
- **Logic Signals**: 3.3V, 5V, 24V DC
- **Sensor Power**: 12V, 24V DC

## Power Supply Design

### Raspberry Pi Power Requirements
```
┌─────────────────────────────────────────────────┐
│              Power Distribution                 │
├─────────────────────────────────────────────────┤
│  24V DC Industrial  →  5V DC Converter  →  Pi   │
│     Supply            (15W minimum)             │
│                                                 │
│  Sensor Power  →  24V DC Distribution           │
│  (Proximity,      (Fused outputs)               │
│   Photoelectric)                                │
│                                                 │
│  Actuator Power → 24V DC Relays/Contactors     │
│  (Lights, Alarms,  (Higher current capacity)   │
│   Reject Mechanisms)                            │
└─────────────────────────────────────────────────┘
```

### Power Supply Selection
```python
# power_calculations.py
class PowerSupplyCalculator:
    """Calculate power supply requirements for vision system"""
    
    def __init__(self):
        self.components = {}
        self.safety_factor = 1.25  # 25% safety margin
    
    def add_component(self, name: str, voltage: float, current_ma: float):
        """Add component to power calculation"""
        self.components[name] = {
            'voltage': voltage,
            'current_ma': current_ma,
            'power_w': voltage * current_ma / 1000
        }
    
    def calculate_requirements(self) -> dict:
        """Calculate total power requirements"""
        total_5v_current = 0
        total_24v_current = 0
        total_power = 0
        
        for name, component in self.components.items():
            voltage = component['voltage']
            current = component['current_ma']
            power = component['power_w']
            
            if voltage == 5.0:
                total_5v_current += current
            elif voltage == 24.0:
                total_24v_current += current
            
            total_power += power
        
        # Apply safety factor
        total_5v_current *= self.safety_factor
        total_24v_current *= self.safety_factor
        total_power *= self.safety_factor
        
        return {
            '5v_current_ma': total_5v_current,
            '24v_current_ma': total_24v_current,
            'total_power_w': total_power,
            'recommended_5v_supply_w': total_5v_current * 5 / 1000,
            'recommended_24v_supply_w': total_24v_current * 24 / 1000
        }

# Example usage
calc = PowerSupplyCalculator()
calc.add_component('Raspberry Pi 4', 5.0, 1200)  # 1.2A at 5V
calc.add_component('Camera Module', 5.0, 300)    # 300mA at 5V
calc.add_component('LED Lighting', 24.0, 2000)   # 2A at 24V
calc.add_component('Proximity Sensors', 24.0, 400) # 400mA at 24V

requirements = calc.calculate_requirements()
print(f"5V Supply needed: {requirements['recommended_5v_supply_w']:.1f}W")
print(f"24V Supply needed: {requirements['recommended_24v_supply_w']:.1f}W")
```

### Industrial Power Supply Wiring
```
┌─────────────────────────────────────────────────┐
│              24V DC Power Supply                │
│  L1 ────────────────── AC Input (120/240V)     │
│  L2/N ──────────────── AC Input                 │
│  PE/GND ────────────── Protective Earth         │
│                                                 │
│  +24V ──────────────── Positive Output          │
│  0V/COM ────────────── Common/Negative          │
│  PE ────────────────── Protective Earth         │
└─────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│            DC/DC Converter (24V → 5V)           │
│  +24V ──────────────── Input Positive           │
│  0V ────────────────── Input Common             │
│                                                 │
│  +5V ───────────────── Output Positive          │
│  0V ────────────────── Output Common            │
└─────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│               Raspberry Pi 4                    │
│  USB-C ─────────────── 5V Power Input           │
│  GND ───────────────── Ground Connection        │
└─────────────────────────────────────────────────┘
```

## GPIO and Digital I/O

### GPIO Pin Configuration
```python
# gpio_interface.py
from gpiozero import LED, Button, DigitalOutputDevice, DigitalInputDevice
import time
import threading

class IndustrialGPIO:
    """Industrial GPIO interface with safety features"""
    
    def __init__(self):
        self.outputs = {}
        self.inputs = {}
        self.monitoring = False
        self.monitor_thread = None
        
    def setup_output(self, name: str, pin: int, initial_state: bool = False):
        """Setup digital output with safety checks"""
        try:
            device = DigitalOutputDevice(pin, initial_value=initial_state)
            self.outputs[name] = {
                'device': device,
                'pin': pin,
                'current_state': initial_state,
                'fault_count': 0
            }
            print(f"Output '{name}' configured on pin {pin}")
        except Exception as e:
            print(f"Error setting up output '{name}': {e}")
    
    def setup_input(self, name: str, pin: int, pull_up: bool = True):
        """Setup digital input with debouncing"""
        try:
            device = DigitalInputDevice(pin, pull_up=pull_up, bounce_time=0.05)
            self.inputs[name] = {
                'device': device,
                'pin': pin,
                'current_state': device.value,
                'last_change': time.time(),
                'change_count': 0
            }
            
            # Setup callback for state changes
            device.when_activated = lambda: self._input_changed(name, True)
            device.when_deactivated = lambda: self._input_changed(name, False)
            
            print(f"Input '{name}' configured on pin {pin}")
        except Exception as e:
            print(f"Error setting up input '{name}': {e}")
    
    def set_output(self, name: str, state: bool) -> bool:
        """Set output state with error checking"""
        if name not in self.outputs:
            print(f"Output '{name}' not configured")
            return False
        
        try:
            output = self.outputs[name]
            output['device'].value = state
            output['current_state'] = state
            return True
        except Exception as e:
            print(f"Error setting output '{name}': {e}")
            self.outputs[name]['fault_count'] += 1
            return False
    
    def get_input(self, name: str) -> bool:
        """Get input state"""
        if name not in self.inputs:
            print(f"Input '{name}' not configured")
            return False
        
        return self.inputs[name]['device'].value
    
    def _input_changed(self, name: str, new_state: bool):
        """Handle input state change"""
        if name in self.inputs:
            input_info = self.inputs[name]
            input_info['current_state'] = new_state
            input_info['last_change'] = time.time()
            input_info['change_count'] += 1
            
            print(f"Input '{name}' changed to {new_state}")
    
    def start_monitoring(self):
        """Start input/output monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor I/O status"""
        while self.monitoring:
            # Check for stuck inputs (no changes for extended period)
            current_time = time.time()
            for name, input_info in self.inputs.items():
                time_since_change = current_time - input_info['last_change']
                if time_since_change > 3600:  # 1 hour without change
                    print(f"Warning: Input '{name}' may be stuck")
            
            # Check output fault counts
            for name, output_info in self.outputs.items():
                if output_info['fault_count'] > 10:
                    print(f"Warning: Output '{name}' has high fault count")
            
            time.sleep(10)  # Check every 10 seconds

# Example usage for vision system
gpio = IndustrialGPIO()

# Setup inputs
gpio.setup_input('part_sensor', 2, pull_up=True)      # Part present sensor
gpio.setup_input('trigger_inspection', 3, pull_up=True) # Inspection trigger
gpio.setup_input('emergency_stop', 4, pull_up=True)    # Emergency stop

# Setup outputs
gpio.setup_output('pass_light', 18, False)            # Green pass light
gpio.setup_output('fail_light', 19, False)            # Red fail light
gpio.setup_output('reject_solenoid', 20, False)       # Part reject mechanism
gpio.setup_output('strobe_light', 21, False)          # Strobe for image capture

gpio.start_monitoring()
```

### Level Shifting for 24V Signals
Many industrial sensors operate at 24V logic levels, requiring level shifting for 3.3V GPIO.

```python
# level_shifting.py
class LevelShifter:
    """Handle 24V to 3.3V level shifting"""
    
    def __init__(self):
        # Using optocouplers for isolation
        self.input_mapping = {
            # 24V input pin -> GPIO pin
            'sensor_1': {'24v_pin': 'X1', 'gpio_pin': 2},
            'sensor_2': {'24v_pin': 'X2', 'gpio_pin': 3},
        }
        
        self.output_mapping = {
            # GPIO pin -> 24V output pin
            'relay_1': {'gpio_pin': 18, '24v_pin': 'Y1'},
            'relay_2': {'gpio_pin': 19, '24v_pin': 'Y2'},
        }
```

### Wiring Diagram - GPIO Interface
```
24V Industrial I/O Module
┌─────────────────────────────────────────────────┐
│                Input Section                    │
│  X1 ──── Proximity Sensor 1 (24V DC)          │
│  X2 ──── Photoelectric Sensor (24V DC)         │
│  X3 ──── Limit Switch (24V DC)                 │
│  COM ─── Common (0V)                           │
│                                                 │
│              Optocouplers                       │
│  X1 ────┐                    ┌──── GPIO 2      │
│         │    ┌─────────┐     │                 │
│         └────┤ ISO     ├─────┘                 │
│              └─────────┘                       │
│                                                 │
│               Output Section                    │
│  Y1 ──── Status Light (24V DC)                │
│  Y2 ──── Reject Solenoid (24V DC)             │
│  COM ─── Common (0V)                           │
│                                                 │
│              Relay Drivers                      │
│  GPIO 18 ────┐                                 │
│              │    ┌─────────┐                  │
│              └────┤ Relay   ├──── Y1           │
│                   └─────────┘                  │
└─────────────────────────────────────────────────┘
```

## Sensor Integration

### Proximity Sensors
Common types used in assembly lines:

#### Inductive Proximity Sensors
```
Specifications:
- Detection Distance: 2-15mm (depending on target)
- Operating Voltage: 10-30V DC
- Output Type: NPN/PNP, NO/NC
- Response Time: <1ms
- Protection: IP67
```

#### Photoelectric Sensors
```python
# photoelectric_sensor.py
class PhotoelectricSensor:
    """Interface for photoelectric sensors"""
    
    def __init__(self, gpio_pin: int, sensor_type: str = 'diffuse'):
        self.gpio_pin = gpio_pin
        self.sensor_type = sensor_type  # 'diffuse', 'through_beam', 'retro_reflective'
        self.gpio = IndustrialGPIO()
        self.gpio.setup_input(f'photo_sensor_{gpio_pin}', gpio_pin)
        
        # Sensor characteristics
        self.detection_range = {
            'diffuse': 200,          # 200mm typical
            'through_beam': 30000,   # 30m typical
            'retro_reflective': 6000  # 6m typical
        }
        
    def is_object_detected(self) -> bool:
        """Check if object is detected"""
        return self.gpio.get_input(f'photo_sensor_{self.gpio_pin}')
    
    def wait_for_object(self, timeout: float = 10.0) -> bool:
        """Wait for object detection with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_object_detected():
                return True
            time.sleep(0.01)  # 10ms polling
        return False

# Wiring diagram for photoelectric sensor
"""
Photoelectric Sensor (PNP Output)
┌─────────────────┐
│  Brown  (+24V)  ├──── +24V Supply
│  Blue   (0V)    ├──── 0V/Common
│  Black  (Out)   ├──── To Optocoupler Input
└─────────────────┘
                           │
                           ▼
Optocoupler Module
┌─────────────────┐        ┌─────────────────┐
│  +24V Input     ├────────┤ Sensor Output   │
│  0V Input       ├────────┤ 0V              │
│                 │        └─────────────────┘
│  +3.3V Output   ├──── GPIO Pin (Raspberry Pi)
│  0V Output      ├──── GND (Raspberry Pi)
└─────────────────┘
"""
```

### Vision Trigger Integration
```python
# vision_trigger.py
import time
import threading
from typing import Callable

class VisionTrigger:
    """Coordinate vision system triggers with sensors"""
    
    def __init__(self):
        self.gpio = IndustrialGPIO()
        self.triggers = {}
        self.active = False
        
    def setup_trigger(self, name: str, sensor_pin: int, 
                     trigger_delay_ms: int = 50,
                     callback: Callable = None):
        """Setup vision trigger based on sensor input"""
        
        self.gpio.setup_input(f'trigger_{name}', sensor_pin)
        
        self.triggers[name] = {
            'sensor_pin': sensor_pin,
            'delay_ms': trigger_delay_ms,
            'callback': callback,
            'last_trigger': 0,
            'trigger_count': 0
        }
        
        # Setup sensor callback
        input_device = self.gpio.inputs[f'trigger_{name}']['device']
        input_device.when_activated = lambda: self._handle_trigger(name)
    
    def _handle_trigger(self, trigger_name: str):
        """Handle trigger event with timing control"""
        current_time = time.time()
        trigger_info = self.triggers[trigger_name]
        
        # Prevent multiple triggers too close together
        time_since_last = (current_time - trigger_info['last_trigger']) * 1000
        if time_since_last < 100:  # 100ms minimum between triggers
            return
        
        trigger_info['last_trigger'] = current_time
        trigger_info['trigger_count'] += 1
        
        # Delay before triggering vision system
        delay_s = trigger_info['delay_ms'] / 1000.0
        
        def delayed_trigger():
            time.sleep(delay_s)
            if trigger_info['callback']:
                trigger_info['callback'](trigger_name)
        
        # Execute trigger in separate thread to avoid blocking
        trigger_thread = threading.Thread(target=delayed_trigger)
        trigger_thread.start()
    
    def get_trigger_statistics(self) -> dict:
        """Get trigger statistics for diagnostics"""
        stats = {}
        for name, trigger_info in self.triggers.items():
            stats[name] = {
                'trigger_count': trigger_info['trigger_count'],
                'last_trigger': trigger_info['last_trigger'],
                'triggers_per_minute': self._calculate_trigger_rate(name)
            }
        return stats
    
    def _calculate_trigger_rate(self, trigger_name: str) -> float:
        """Calculate triggers per minute"""
        # This would maintain a sliding window of trigger times
        # For simplicity, returning a placeholder
        return 0.0

# Example integration with vision system
def vision_inspection_callback(trigger_name: str):
    """Callback function for vision inspection"""
    print(f"Vision inspection triggered by {trigger_name}")
    # Trigger camera capture and processing
    # Send results to PLC
    
trigger_system = VisionTrigger()
trigger_system.setup_trigger(
    'part_present',
    sensor_pin=2,
    trigger_delay_ms=100,  # 100ms delay for part to settle
    callback=vision_inspection_callback
)
```

## Actuator Control

### Relay Control for Industrial Outputs
```python
# relay_control.py
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class RelayConfig:
    """Configuration for relay output"""
    name: str
    gpio_pin: int
    normally_open: bool = True
    max_on_time: float = 60.0  # Maximum continuous on time (safety)
    min_off_time: float = 0.1  # Minimum off time between activations

class RelayController:
    """Control industrial relays with safety features"""
    
    def __init__(self):
        self.relays: Dict[str, dict] = {}
        self.gpio = IndustrialGPIO()
        
    def add_relay(self, config: RelayConfig):
        """Add relay to controller"""
        self.gpio.setup_output(config.name, config.gpio_pin, False)
        
        self.relays[config.name] = {
            'config': config,
            'state': False,
            'turn_on_time': 0,
            'turn_off_time': time.time(),
            'activation_count': 0,
            'total_on_time': 0
        }
    
    def activate_relay(self, name: str, duration: float = None) -> bool:
        """Activate relay with safety checks"""
        if name not in self.relays:
            print(f"Relay '{name}' not configured")
            return False
        
        relay_info = self.relays[name]
        config = relay_info['config']
        current_time = time.time()
        
        # Check minimum off time
        time_since_off = current_time - relay_info['turn_off_time']
        if time_since_off < config.min_off_time:
            print(f"Relay '{name}': Minimum off time not met")
            return False
        
        # Check if already active
        if relay_info['state']:
            print(f"Relay '{name}' already active")
            return False
        
        # Activate relay
        success = self.gpio.set_output(name, True)
        if success:
            relay_info['state'] = True
            relay_info['turn_on_time'] = current_time
            relay_info['activation_count'] += 1
            
            # Set up automatic deactivation if duration specified
            if duration:
                threading.Timer(duration, self._auto_deactivate, args=[name]).start()
            
            print(f"Relay '{name}' activated")
        
        return success
    
    def deactivate_relay(self, name: str) -> bool:
        """Deactivate relay"""
        if name not in self.relays:
            print(f"Relay '{name}' not configured")
            return False
        
        relay_info = self.relays[name]
        current_time = time.time()
        
        if not relay_info['state']:
            print(f"Relay '{name}' already inactive")
            return True
        
        # Deactivate relay
        success = self.gpio.set_output(name, False)
        if success:
            relay_info['state'] = False
            relay_info['turn_off_time'] = current_time
            
            # Update total on time
            on_duration = current_time - relay_info['turn_on_time']
            relay_info['total_on_time'] += on_duration
            
            print(f"Relay '{name}' deactivated")
        
        return success
    
    def _auto_deactivate(self, name: str):
        """Automatically deactivate relay after timeout"""
        self.deactivate_relay(name)
    
    def emergency_stop_all(self):
        """Emergency stop - deactivate all relays"""
        print("EMERGENCY STOP - Deactivating all relays")
        for name in self.relays.keys():
            self.gpio.set_output(name, False)
            self.relays[name]['state'] = False
    
    def get_relay_status(self) -> Dict:
        """Get status of all relays"""
        status = {}
        current_time = time.time()
        
        for name, relay_info in self.relays.items():
            config = relay_info['config']
            
            # Calculate current on time if active
            current_on_time = 0
            if relay_info['state']:
                current_on_time = current_time - relay_info['turn_on_time']
                
                # Check for safety timeout
                if current_on_time > config.max_on_time:
                    print(f"WARNING: Relay '{name}' exceeded maximum on time")
                    self.deactivate_relay(name)
            
            status[name] = {
                'active': relay_info['state'],
                'current_on_time': current_on_time,
                'total_on_time': relay_info['total_on_time'],
                'activation_count': relay_info['activation_count']
            }
        
        return status

# Example setup for vision system outputs
relay_controller = RelayController()

# Add relays for vision system
relay_controller.add_relay(RelayConfig(
    name='pass_light',
    gpio_pin=18,
    max_on_time=300.0  # 5 minutes max
))

relay_controller.add_relay(RelayConfig(
    name='fail_light',
    gpio_pin=19,
    max_on_time=300.0
))

relay_controller.add_relay(RelayConfig(
    name='reject_solenoid',
    gpio_pin=20,
    max_on_time=2.0,   # 2 seconds max for solenoid
    min_off_time=0.5   # 500ms minimum between activations
))

relay_controller.add_relay(RelayConfig(
    name='alarm_horn',
    gpio_pin=21,
    max_on_time=10.0   # 10 seconds max for alarm
))
```

### Solenoid Valve Control
```python
# solenoid_control.py
class SolenoidController:
    """Control pneumatic solenoid valves"""
    
    def __init__(self, relay_controller: RelayController):
        self.relay_controller = relay_controller
        self.solenoids = {}
    
    def add_solenoid(self, name: str, relay_name: str, 
                    actuation_time: float = 0.5):
        """Add solenoid valve configuration"""
        self.solenoids[name] = {
            'relay_name': relay_name,
            'actuation_time': actuation_time,
            'cycle_count': 0
        }
    
    def actuate_solenoid(self, name: str) -> bool:
        """Actuate solenoid for specified time"""
        if name not in self.solenoids:
            print(f"Solenoid '{name}' not configured")
            return False
        
        solenoid_info = self.solenoids[name]
        relay_name = solenoid_info['relay_name']
        actuation_time = solenoid_info['actuation_time']
        
        # Activate relay for specified duration
        success = self.relay_controller.activate_relay(relay_name, actuation_time)
        
        if success:
            solenoid_info['cycle_count'] += 1
            print(f"Solenoid '{name}' actuated for {actuation_time}s")
        
        return success

# Example usage for part rejection
solenoid_controller = SolenoidController(relay_controller)
solenoid_controller.add_solenoid('reject_mechanism', 'reject_solenoid', 0.3)

# Reject failed parts
def reject_part():
    solenoid_controller.actuate_solenoid('reject_mechanism')
```

## Industrial Communication

### Ethernet/IP Integration
```python
# ethernet_ip.py
import socket
import struct
import threading
import time

class EthernetIPClient:
    """Basic Ethernet/IP communication client"""
    
    def __init__(self, plc_ip: str, plc_port: int = 44818):
        self.plc_ip = plc_ip
        self.plc_port = plc_port
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to PLC via Ethernet/IP"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.plc_ip, self.plc_port))
            self.connected = True
            print(f"Connected to PLC at {self.plc_ip}:{self.plc_port}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from PLC"""
        if self.socket:
            self.socket.close()
            self.connected = False
    
    def read_tag(self, tag_name: str):
        """Read tag value from PLC"""
        # Simplified implementation
        # Real implementation would use CIP protocol
        pass
    
    def write_tag(self, tag_name: str, value):
        """Write tag value to PLC"""
        # Simplified implementation
        pass

class HMIInterface:
    """Human Machine Interface for vision system"""
    
    def __init__(self):
        self.system_status = {
            'running': False,
            'parts_inspected': 0,
            'parts_passed': 0,
            'parts_failed': 0,
            'current_mode': 'manual',
            'last_error': None
        }
        
        self.alarms = []
        
    def update_production_count(self, passed: bool):
        """Update production counters"""
        self.system_status['parts_inspected'] += 1
        if passed:
            self.system_status['parts_passed'] += 1
        else:
            self.system_status['parts_failed'] += 1
    
    def add_alarm(self, alarm_type: str, message: str):
        """Add alarm to system"""
        alarm = {
            'timestamp': time.time(),
            'type': alarm_type,
            'message': message,
            'acknowledged': False
        }
        self.alarms.append(alarm)
        print(f"ALARM [{alarm_type}]: {message}")
    
    def acknowledge_alarm(self, alarm_index: int):
        """Acknowledge alarm"""
        if 0 <= alarm_index < len(self.alarms):
            self.alarms[alarm_index]['acknowledged'] = True
    
    def get_system_status(self) -> dict:
        """Get current system status for HMI"""
        pass_rate = 0
        if self.system_status['parts_inspected'] > 0:
            pass_rate = (self.system_status['parts_passed'] / 
                        self.system_status['parts_inspected']) * 100
        
        return {
            **self.system_status,
            'pass_rate': pass_rate,
            'active_alarms': [a for a in self.alarms if not a['acknowledged']]
        }
```

## Grounding and Shielding

### Proper Grounding Techniques
```
Industrial Grounding Hierarchy:
┌─────────────────────────────────────────────────┐
│  1. Safety/Protective Earth (PE)               │
│     - Connect all metal enclosures              │
│     - Use green/yellow wire (IEC) or green (US) │
│                                                 │
│  2. Functional/Signal Ground                    │
│     - Common reference for signals              │
│     - Keep separate from safety ground          │
│                                                 │
│  3. Noise/Shield Ground                         │
│     - Cable shields and EMI protection          │
│     - Connect at one end only (avoid loops)     │
└─────────────────────────────────────────────────┘
```

### EMI/EMC Considerations
```python
# emi_protection.py
class EMIProtection:
    """Guidelines for EMI/EMC protection"""
    
    @staticmethod
    def cable_routing_guidelines():
        """Best practices for cable routing"""
        guidelines = [
            "Separate power and signal cables by minimum 300mm",
            "Use twisted pair cables for differential signals",
            "Route cables perpendicular when crossing unavoidable",
            "Use metal conduit for high-noise environments",
            "Ground cable shields at one end only",
            "Avoid routing near motors, drives, and welders"
        ]
        return guidelines
    
    @staticmethod
    def filter_recommendations():
        """Power line filter recommendations"""
        return {
            'input_filter': 'Use line filters on all AC inputs',
            'dc_filters': 'Add ferrite beads on DC power lines',
            'signal_filters': 'Use RC filters on GPIO inputs',
            'ground_loops': 'Use isolation transformers if needed'
        }

# Grounding configuration
grounding_config = {
    'safety_ground': {
        'description': 'Protective earth connection',
        'wire_color': 'green_yellow',
        'connection_points': ['enclosure', 'power_supply', 'devices']
    },
    'signal_ground': {
        'description': 'Signal reference ground',
        'wire_color': 'black',
        'connection_points': ['raspberry_pi', 'sensors', 'io_modules']
    },
    'shield_ground': {
        'description': 'Cable shield termination',
        'connection': 'one_end_only',
        'termination_point': 'control_panel'
    }
}
```

### Electrical Panel Layout
```
Typical Control Panel Layout:
┌─────────────────────────────────────────────────┐
│                Main Breaker                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │   Fuse  │  │   Fuse  │  │   Fuse  │         │
│  │  Block  │  │  Block  │  │  Block  │         │
│  │ 24V DC  │  │ Control │  │ Lights  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │           Power Supplies                │   │
│  │  ┌─────────┐    ┌─────────────────┐    │   │
│  │  │ 24V DC  │    │   DC/DC Conv    │    │   │
│  │  │ 5A      │    │   24V → 5V      │    │   │
│  │  └─────────┘    └─────────────────┘    │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │              I/O Modules               │   │
│  │  ┌─────────┐    ┌─────────────────┐    │   │
│  │  │ Digital │    │     Relay       │    │   │
│  │  │ Input   │    │    Output       │    │   │
│  │  │ 24V DC  │    │    Module       │    │   │
│  │  └─────────┘    └─────────────────┘    │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │           Raspberry Pi                  │   │
│  │  ┌─────────────────────────────────┐    │   │
│  │  │        DIN Rail Mount           │    │   │
│  │  │         Enclosure               │    │   │
│  │  └─────────────────────────────────┘    │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Troubleshooting

### Common Electrical Issues
```python
# troubleshooting.py
class ElectricalDiagnostics:
    """Diagnostic tools for electrical issues"""
    
    def __init__(self):
        self.test_results = {}
    
    def voltage_test(self, test_point: str, expected_voltage: float, 
                    measured_voltage: float, tolerance: float = 0.1):
        """Test voltage levels"""
        deviation = abs(measured_voltage - expected_voltage)
        within_tolerance = deviation <= tolerance
        
        result = {
            'test_point': test_point,
            'expected': expected_voltage,
            'measured': measured_voltage,
            'deviation': deviation,
            'pass': within_tolerance
        }
        
        self.test_results[test_point] = result
        
        if not within_tolerance:
            print(f"VOLTAGE FAULT at {test_point}: "
                  f"Expected {expected_voltage}V, got {measured_voltage}V")
        
        return within_tolerance
    
    def continuity_test(self, circuit_name: str, has_continuity: bool, 
                       expected_continuity: bool = True):
        """Test circuit continuity"""
        test_pass = has_continuity == expected_continuity
        
        result = {
            'circuit': circuit_name,
            'has_continuity': has_continuity,
            'expected_continuity': expected_continuity,
            'pass': test_pass
        }
        
        self.test_results[f"{circuit_name}_continuity"] = result
        
        if not test_pass:
            status = "OPEN" if not has_continuity else "SHORT"
            print(f"CONTINUITY FAULT in {circuit_name}: Circuit {status}")
        
        return test_pass
    
    def insulation_test(self, circuit_name: str, insulation_resistance: float,
                       minimum_resistance: float = 1000000):  # 1MΩ minimum
        """Test insulation resistance"""
        test_pass = insulation_resistance >= minimum_resistance
        
        result = {
            'circuit': circuit_name,
            'resistance': insulation_resistance,
            'minimum': minimum_resistance,
            'pass': test_pass
        }
        
        self.test_results[f"{circuit_name}_insulation"] = result
        
        if not test_pass:
            print(f"INSULATION FAULT in {circuit_name}: "
                  f"Resistance {insulation_resistance}Ω (min {minimum_resistance}Ω)")
        
        return test_pass
    
    def generate_test_report(self) -> str:
        """Generate diagnostic test report"""
        report = "Electrical Diagnostic Report\n"
        report += "=" * 40 + "\n\n"
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['pass'])
        
        report += f"Summary: {passed_tests}/{total_tests} tests passed\n\n"
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result['pass'] else "FAIL"
            report += f"{test_name}: {status}\n"
        
        return report

# Common troubleshooting procedures
def troubleshoot_gpio_issue(gpio_pin: int):
    """Troubleshoot GPIO pin issues"""
    diagnostics = ElectricalDiagnostics()
    
    print(f"Troubleshooting GPIO pin {gpio_pin}")
    
    # Check if pin is configured correctly
    # Check voltage levels
    # Check for shorts or opens
    # Check for noise/interference
    
    return diagnostics.generate_test_report()

def troubleshoot_sensor_issue(sensor_name: str):
    """Troubleshoot sensor connectivity issues"""
    print(f"Troubleshooting sensor: {sensor_name}")
    
    troubleshooting_steps = [
        "1. Check power supply voltage (24V DC ±10%)",
        "2. Verify wiring connections (tight, no corrosion)",
        "3. Test sensor output with multimeter",
        "4. Check for proper grounding",
        "5. Verify sensor mounting and alignment",
        "6. Test with known good sensor if available"
    ]
    
    for step in troubleshooting_steps:
        print(step)
    
    return troubleshooting_steps

def troubleshoot_communication_issue():
    """Troubleshoot PLC communication issues"""
    print("Troubleshooting PLC communication")
    
    checks = [
        "Network connectivity (ping test)",
        "IP address configuration",
        "Port accessibility (firewall)",
        "Protocol compatibility",
        "Cable integrity",
        "Network switch/hub operation"
    ]
    
    return checks
```

### Test Procedures
```python
# test_procedures.py
import time

class SystemTestProcedures:
    """Automated test procedures for vision system"""
    
    def __init__(self, gpio: IndustrialGPIO, relay_controller: RelayController):
        self.gpio = gpio
        self.relay_controller = relay_controller
        self.test_results = []
    
    def power_on_test(self) -> bool:
        """Power-on self test"""
        print("Starting Power-On Self Test (POST)")
        
        tests = [
            self._test_power_supplies,
            self._test_gpio_outputs,
            self._test_gpio_inputs,
            self._test_communication,
            self._test_camera_connection
        ]
        
        all_passed = True
        for test in tests:
            try:
                result = test()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"Test failed with exception: {e}")
                all_passed = False
        
        print(f"POST Result: {'PASS' if all_passed else 'FAIL'}")
        return all_passed
    
    def _test_power_supplies(self) -> bool:
        """Test power supply voltages"""
        print("Testing power supplies...")
        # Implementation would measure actual voltages
        # For demonstration, assume test passes
        time.sleep(1)
        print("Power supplies: OK")
        return True
    
    def _test_gpio_outputs(self) -> bool:
        """Test all GPIO outputs"""
        print("Testing GPIO outputs...")
        
        # Test each relay output
        for relay_name in self.relay_controller.relays.keys():
            print(f"Testing relay: {relay_name}")
            
            # Activate relay briefly
            self.relay_controller.activate_relay(relay_name, 0.5)
            time.sleep(0.6)
            
            # Check if relay deactivated properly
            status = self.relay_controller.get_relay_status()
            if status[relay_name]['active']:
                print(f"ERROR: Relay {relay_name} failed to deactivate")
                return False
        
        print("GPIO outputs: OK")
        return True
    
    def _test_gpio_inputs(self) -> bool:
        """Test GPIO inputs"""
        print("Testing GPIO inputs...")
        print("Please activate each sensor input for testing...")
        
        # In a real test, this would systematically test each input
        # For demonstration, assume test passes
        time.sleep(2)
        print("GPIO inputs: OK")
        return True
    
    def _test_communication(self) -> bool:
        """Test communication interfaces"""
        print("Testing communication...")
        # Test network connectivity, PLC communication, etc.
        time.sleep(1)
        print("Communication: OK")
        return True
    
    def _test_camera_connection(self) -> bool:
        """Test camera connection and functionality"""
        print("Testing camera connection...")
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.start()
            time.sleep(1)
            frame = picam2.capture_array()
            picam2.stop()
            
            if frame is not None and frame.size > 0:
                print("Camera: OK")
                return True
            else:
                print("ERROR: Camera returned invalid frame")
                return False
                
        except Exception as e:
            print(f"ERROR: Camera test failed: {e}")
            return False
    
    def functional_test(self) -> bool:
        """Complete functional test sequence"""
        print("Starting Functional Test")
        
        # Test complete vision system workflow
        test_sequence = [
            "Trigger part present sensor",
            "Capture image",
            "Process image",
            "Generate result",
            "Activate output (pass/fail light)",
            "Log result"
        ]
        
        for step in test_sequence:
            print(f"Testing: {step}")
            time.sleep(1)  # Simulate test time
        
        print("Functional test: COMPLETE")
        return True

# Example usage
# gpio = IndustrialGPIO()
# relay_controller = RelayController()
# test_system = SystemTestProcedures(gpio, relay_controller)
# test_system.power_on_test()
```

## Practical Exercises

### Exercise 1: Basic Wiring Setup
1. Wire a 24V DC power supply to Raspberry Pi via DC/DC converter
2. Connect proximity sensor through optocoupler
3. Connect relay output for indicator light
4. Test basic I/O functionality

### Exercise 2: Sensor Integration
1. Connect multiple sensor types (proximity, photoelectric)
2. Implement sensor monitoring and diagnostics
3. Add trigger logic for vision system
4. Test sensor response times

### Exercise 3: Actuator Control
1. Wire solenoid valve through relay
2. Implement safety interlocks
3. Add manual override controls
4. Test emergency stop functionality

### Exercise 4: System Integration
1. Complete electrical panel assembly
2. Implement PLC communication
3. Add HMI interface
4. Perform full system testing

## Safety Checklist

### Pre-Installation Safety
- [ ] Review electrical drawings and specifications
- [ ] Verify all components are rated for application
- [ ] Check local electrical codes and standards
- [ ] Obtain proper permits if required
- [ ] Plan emergency shutdown procedures

### Installation Safety
- [ ] Use lockout/tagout procedures
- [ ] Verify circuits are de-energized before work
- [ ] Use appropriate PPE
- [ ] Follow proper grounding procedures
- [ ] Test all safety systems before operation

### Operational Safety
- [ ] Train operators on safety procedures
- [ ] Implement regular inspection schedule
- [ ] Maintain emergency contact information
- [ ] Keep electrical drawings current
- [ ] Document all modifications

## References

1. [IEC 61131 - Programmable Controllers](https://www.iec.ch/dyn/www/f?p=103:23:0::::FSP_ORG_ID,FSP_LANG_ID:1,25)
2. [NEMA Standards Publications](https://www.nema.org/Standards/CompleteStandardsLibrary)
3. [Industrial Wiring Best Practices](https://www.automationdirect.com/static/manuals/wiring_manual.pdf)
4. [Raspberry Pi GPIO Documentation](https://www.raspberrypi.org/documentation/usage/gpio/)

---

**Project Complete**: Return to [Main README](../README.md)