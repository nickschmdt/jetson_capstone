import time
from camera.image_capture import capture_from_csi # run image capturing function
from detection.object_detection import detect_objects # still need to create
from sensors.ultrasonic_sensor_jetson import read_distance # read distances from sensor
from feedback.haptic import vibrate # still need to create haptic code
from feedback.tts import speak # still need to create tts code

def main():
    print("System starting up...")
    time.sleep(2)
    print("Camera initialized. Starting detection loop...")

    last_tts_time = 0
    tts_interval = 15  # seconds

    while True:
        # capture frame from camera
        frame = capture_from_csi()

        # run object detection (using pretrained TFLite model)
        detected_objects = detect_objects(frame)

        # read distance from ultrasonic sensor
        distance = read_distance()

        # decision logic
        if distance < 50:  # example threshold
            vibrate(True)  # trigger haptic feedback
        else:
            vibrate(False)

        # every 15 seconds, speak out detected objects + distance
        current_time = time.time()
        if current_time - last_tts_time > tts_interval:
            objects_text = ", ".join(detected_objects) if detected_objects else "nothing detected"
            speak(f"I see {objects_text} approximately {distance:.1f} centimeters away.")
            last_tts_time = current_time

        time.sleep(0.1)

if __name__ == "__main__":
    main()
