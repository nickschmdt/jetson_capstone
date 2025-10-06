import serial
import time

# Replace with the actual port your Arduino uses:
# On Windows: "COM3" or similar
# On Jetson/Linux: "/dev/ttyACM0" or "/dev/ttyUSB0"
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # wait for connection to initialize

print("Reading distances...")

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line:
            print(f"Distance: {line} cm")