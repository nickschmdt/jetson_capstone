# code for capturing video feed with CSI camera

import cv2


#defining pipeline for opencv2 to communicate with camera 
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


#capture function
def capture_from_csi():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    #check if camera opens
    if not cap.isOpened():
        print("unable to open CSI camera")
        return

    print("CSI camera opened press q to quit.")

    #main frame capturing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break

        # show captured frames
        cv2.imshow("CSI Camera", frame)

        #press q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #stops opencv comms with camera
    cap.release()
    #closes video window
    cv2.destroyAllWindows()

