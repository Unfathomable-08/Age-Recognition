import cv2
import os
from datetime import datetime

def capture_image(save_dir = "captures"):
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press SPACE to capture image, or ESC to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(100)

        if key % 256 == 27:  # ESC pressed
            print("Closing without saving.")
            break
        elif key % 256 == 32:  # SPACE pressed
            # Generate filename using current time
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(save_dir, f"{timestamp}.jpg")

            cv2.imwrite(save_path, frame)
            print(f"Image saved as {save_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

capture_image()
