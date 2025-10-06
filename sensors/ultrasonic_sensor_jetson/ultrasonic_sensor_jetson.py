import serial
import time

SERIAL_PORT = "COM3"  
BAUD_RATE = 9600  

def read_distances():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # allow Arduino to reset

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip() # converts bytes into string
                try:
                    distance = float(line)
                    print(f"Distance: {distance} cm")
                except ValueError:
                    pass  # ignore garbage lines

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    read_distances()