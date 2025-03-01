import gradio as gr
from app.processing.image_processor import process_image
from app.config import CUSTOM_TEMP_DIR

def create_interface():
    return gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="filepath", label="Upload Image"),
            gr.Textbox(label="Enter Prompt", placeholder="Describe this image.")
        ],
        outputs=gr.Textbox(label="Model Output"),
        title="Qwen-VL Image Description",
        description="Upload an image and enter a prompt to get a description from the Qwen-VL model.",
        examples=[
            ["path/to/example/image1.jpg", "Describe this image."],
            ["path/to/example/image2.jpg", "What's happening in this picture?"]
        ],
        cache_examples=True,
        tempdir=CUSTOM_TEMP_DIR
    )
