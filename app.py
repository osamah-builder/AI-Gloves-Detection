from ultralytics import YOLO
import gradio as gr
import cv2

# تحميل المودل
model = YOLO("yolov8n.pt")  # أو best.pt إذا كان لديك مودل مدرب

def detect(image):
    results = model(image)
    result_image = results[0].plot()
    return result_image

interface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="AI Gloves Detection",
    description="Upload an image and detect gloves using YOLOv8"
)

interface.launch()