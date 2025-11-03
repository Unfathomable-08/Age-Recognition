import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from model import CNN  

# =============== LOAD MODEL ===============
model = CNN(num_classes=8)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

classes = ["01-10", "21-30", "11-20", "41-55", "31-40", "56-65", "66-80", "88+"]

# =============== PREPROCESS ===============
def preprocess_image(image):
    img = Image.fromarray(image).convert('L')
    img = img.resize((200, 200))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,200,200)
    return img

# =============== PREDICT FUNCTION ===============
def predict_frame(frame):
    if frame is None:
        return frame, "No frame"

    # Resize for display (optional)
    display_frame = frame.copy()

    # Preprocess + predict
    input_tensor = preprocess_image(frame)
    with torch.no_grad():
        outputs = model(input_tensor)
        prob = torch.softmax(outputs, dim=1)
        confidence = prob.max().item()
        _, predicted = torch.max(outputs, 1)
        age_range = classes[predicted.item()]

    # Draw prediction on frame
    text = f"{age_range} ({confidence:.2f})"
    cv2.putText(display_frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return display_frame, text

# =============== LIVE VIDEO FEED ===============
def video_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for Gradio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict
        annotated_frame, prediction_text = predict_frame(frame_rgb)

        yield annotated_frame, prediction_text

    cap.release()

# =============== GRADIO UI ===============
with gr.Blocks(title="Live Age Detector") as demo:
    gr.Markdown("# Live Age Prediction from Webcam")
    gr.Markdown("Point camera at face → see age range in real time!")

    with gr.Row():
        with gr.Column(scale=3):
            video = gr.Image(label="Live Feed", streaming=True, height=480)
        with gr.Column(scale=1):
            result = gr.Textbox(label="Predicted Age", lines=2)

    gr.Markdown("### Tip: Center face in frame for best results")

    # Auto-start on load
    demo.load(
        fn=video_feed,
        outputs=[video, result]
    )

# =============== LAUNCH ===============
demo.launch(
    share=True,        # Get public link
    server_port=7860,
    height=600
)
