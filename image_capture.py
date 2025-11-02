import cv2
from PIL import Image
from model import CNN
import torch
import numpy as np

model = CNN(num_classes=8)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

classes = ["01-10", "21-30", "11-20", "41-55", "31-40", "56-65", "66-80", "88+"]

def preprocess_image(image):
    image = image.convert('L')  # grayscale
    image = image.resize((200, 200))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 200, 200]
    return image

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
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess_image(pil_img)

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                print(outputs, predicted)
                print(classes[predicted.item()])

            break

    cap.release()
    cv2.destroyAllWindows()

capture_image()
