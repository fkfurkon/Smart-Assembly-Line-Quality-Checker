# Raspberry Pi Single Board Computer Structure

## Overview
The Raspberry Pi is a credit-card-sized single-board computer that provides a complete computing solution in a compact form factor. This section covers the hardware architecture, components, and design principles.

## Table of Contents
1. [Hardware Architecture](#hardware-architecture)
2. [Key Components](#key-components)
3. [GPIO Interface](#gpio-interface)
4. [Power System](#power-system)
5. [Connectivity Options](#connectivity-options)
6. [Performance Specifications](#performance-specifications)

## Hardware Architecture

### System-on-Chip (SoC)
The Raspberry Pi 4 uses the Broadcom BCM2711 SoC which includes:
- **CPU**: Quad-core ARM Cortex-A72 (ARM v8) 64-bit @ 1.5GHz
- **GPU**: VideoCore VI supporting OpenGL ES 3.x
- **RAM**: 2GB, 4GB, or 8GB LPDDR4-3200 SDRAM
- **Storage**: MicroSD card slot (supports up to 1TB)

### Block Diagram
```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi 4                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   BCM2711    │  │    RAM      │  │   Power Mgmt    │ │
│  │     SoC      │  │ 2/4/8 GB    │  │     PMIC        │ │
│  │              │  │   LPDDR4    │  │                 │ │
│  └──────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │     GPIO     │  │    Camera   │  │   Display       │ │
│  │   40-pin     │  │   Interface │  │   Interface     │ │
│  │   Header     │  │    (CSI)    │  │    (DSI)        │ │
│  └──────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Ethernet   │  │    USB      │  │     Audio       │ │
│  │  Gigabit     │  │  2x USB3    │  │   3.5mm Jack    │ │
│  │    Port      │  │  2x USB2    │  │     + HDMI      │ │
│  └──────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Central Processing Unit (CPU)
- **Architecture**: ARM Cortex-A72 (64-bit)
- **Cores**: 4 cores
- **Clock Speed**: 1.5 GHz (can be overclocked)
- **Instruction Set**: ARMv8-A
- **Cache**: 32KB L1 I-cache, 32KB L1 D-cache, 1MB L2 cache

### 2. Graphics Processing Unit (GPU)
- **Model**: VideoCore VI
- **Features**:
  - Hardware accelerated video decode/encode
  - OpenGL ES 3.x support
  - Vulkan 1.0 support
  - 4Kp60 HEVC decode capability

### 3. Memory System
- **Type**: LPDDR4-3200 SDRAM
- **Capacity**: Available in 2GB, 4GB, 8GB variants
- **Bandwidth**: 25.6 GB/s theoretical peak

### 4. Storage Interface
- **Primary**: MicroSD card slot
- **Boot**: Supports USB boot and network boot
- **External**: USB storage devices

## GPIO Interface

### 40-Pin GPIO Header
The GPIO (General Purpose Input/Output) header provides:
- 26 GPIO pins
- 2x 5V power pins
- 2x 3.3V power pins
- 8x Ground pins
- 2x ID EEPROM pins

### GPIO Pin Layout
```
    3V3  (1) (2)  5V
  GPIO2  (3) (4)  5V
  GPIO3  (5) (6)  GND
  GPIO4  (7) (8)  GPIO14
    GND  (9) (10) GPIO15
 GPIO17 (11) (12) GPIO18
 GPIO27 (13) (14) GND
 GPIO22 (15) (16) GPIO23
    3V3 (17) (18) GPIO24
 GPIO10 (19) (20) GND
  GPIO9 (21) (22) GPIO25
 GPIO11 (23) (24) GPIO8
    GND (25) (26) GPIO7
  GPIO0 (27) (28) GPIO1
  GPIO5 (29) (30) GND
  GPIO6 (31) (32) GPIO12
 GPIO13 (33) (34) GND
 GPIO19 (35) (36) GPIO16
 GPIO26 (37) (38) GPIO20
    GND (39) (40) GPIO21
```

### GPIO Capabilities
- **Digital I/O**: 3.3V logic levels
- **PWM**: 2 hardware PWM channels
- **SPI**: 2 SPI interfaces
- **I2C**: 1 I2C interface
- **UART**: 1 UART interface

## Power System

### Power Requirements
- **Input Voltage**: 5V ±0.25V
- **Current**: 3A recommended (15W)
- **Connector**: USB-C (Pi 4) or Micro-USB (earlier models)

### Power Distribution
```
5V Input → Power Management IC → 3.3V Rail → GPIO/Peripherals
                               → 1.8V Rail → SoC Core
                               → 1.2V Rail → RAM
```

### Power Consumption
- **Idle**: ~2.7W
- **Stress Test**: ~7.6W
- **Typical Usage**: 4-6W

## Connectivity Options

### Wired Connectivity
1. **Ethernet**: Gigabit Ethernet (1000 Mbps)
2. **USB**: 2x USB 3.0, 2x USB 2.0 ports
3. **Display**: 2x micro HDMI ports (4K60 support)
4. **Audio**: 3.5mm analog audio/video jack

### Wireless Connectivity
1. **Wi-Fi**: 802.11ac dual-band (2.4/5 GHz)
2. **Bluetooth**: Bluetooth 5.0, Bluetooth Low Energy (BLE)

### Camera and Display Interfaces
1. **Camera Serial Interface (CSI)**: For Raspberry Pi cameras
2. **Display Serial Interface (DSI)**: For touch displays

## Performance Specifications

### Raspberry Pi 4 Model B Performance
| Component | Specification |
|-----------|--------------|
| CPU Cores | 4x ARM Cortex-A72 @ 1.5GHz |
| GPU | VideoCore VI @ 500MHz |
| RAM | 2GB/4GB/8GB LPDDR4-3200 |
| Storage | MicroSD, USB 3.0 |
| Network | Gigabit Ethernet, 802.11ac Wi-Fi |
| Video | 2x 4K60 or 1x 4K60 + 1x 1080p60 |
| I/O | 40-pin GPIO, 4x USB, 2x HDMI |

### Benchmark Comparisons
- **CPU Performance**: ~4x faster than Raspberry Pi 3B+
- **Memory Bandwidth**: ~3x improvement over Pi 3B+
- **Network Throughput**: Full gigabit speeds (vs 330Mbps on 3B+)

## Applications in Industrial Settings

### Advantages for Assembly Line Vision
1. **Compact Form Factor**: Easy integration into tight spaces
2. **Low Power Consumption**: Suitable for continuous operation
3. **GPIO Flexibility**: Direct sensor/actuator control
4. **Cost Effective**: Affordable for multiple deployment points
5. **Community Support**: Extensive documentation and libraries

### Limitations to Consider
1. **Processing Power**: Limited compared to full PCs
2. **Real-time Constraints**: Soft real-time system
3. **Industrial Durability**: May need protection in harsh environments
4. **Heat Management**: Consider cooling in enclosed spaces

## Practical Exercises

### Exercise 1: Hardware Identification
1. Obtain a Raspberry Pi 4
2. Identify all major components on the board
3. Locate the GPIO header and identify pin 1
4. Find the camera and display connectors

### Exercise 2: Power Analysis
1. Measure idle power consumption
2. Test power consumption under CPU load
3. Observe thermal behavior during stress testing

### Exercise 3: GPIO Exploration
1. Use a multimeter to verify GPIO voltage levels
2. Test digital output functionality
3. Measure PWM signal characteristics

## References and Further Reading

1. [Official Raspberry Pi Documentation](https://www.raspberrypi.org/documentation/)
2. [BCM2711 ARM Processor Datasheet](https://datasheets.raspberrypi.org/bcm2711/bcm2711-peripherals.pdf)
3. [GPIO Programming Guide](https://www.raspberrypi.org/documentation/usage/gpio/)
4. [Hardware Design Guidelines](https://www.raspberrypi.org/documentation/hardware/raspberrypi/)

---

**Next Section**: [Raspberry Pi Software →](../02-raspberry-pi-software/README.md)