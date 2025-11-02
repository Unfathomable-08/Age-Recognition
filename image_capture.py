import cv2
from PIL import Image
import torch
import numpy as np
import gradio as gr
from model import CNN

# Load model
model = CNN(num_classes=8)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

classes = ["01-10", "21-30", "11-20", "41-55", "31-40", "56-65", "66-80", "88+"]

def preprocess_image(image):
    image = Image.fromarray(image).convert('L')
    image = image.resize((200, 200))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    return image

def predict_age(image):
    if image is None:
        return "No image"

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        age_range = classes[predicted.item()]
    return age_range

# Create Gradio interface
iface = gr.Interface(
    fn=predict_age,
    inputs=gr.Image(label="Upload or Use Webcam", tool="webcam"),
    outputs=gr.Textbox(label="Predicted Age Range"),
    title="Age Detector",
    description="Click the webcam button to capture your face and predict age!",
    allow_flagging="never"
)

# Launch
iface.launch()