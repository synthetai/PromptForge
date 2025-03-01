import gradio as gr
from app.processing.image_processor import process_image

def create_interface():
    def handle_image_and_prompt(image, prompt, target_language):
        if image is None:
            return "Please upload an image."
        enhanced_prompt = process_image(image, prompt, target_language)
        return enhanced_prompt

    with gr.Blocks(title="PromptForge") as interface:
        gr.Markdown("# PromptForge")
        gr.Markdown("Upload an image, enter a prompt, and select the target language for prompt enhancement.")
        
        image_input = gr.Image(type="filepath", label="Upload Image")
        prompt_input = gr.Textbox(label="Prompt", placeholder="Describe this image.")
        language_input = gr.Radio(["CH", "EN"], label="Target language of prompt enhance", value="CH")
        
        output = gr.Textbox(label="Prompt Enhance Output")
        
        gr.Interface(
            fn=handle_image_and_prompt,
            inputs=[image_input, prompt_input, language_input],
            outputs=output,
            live=False  # 这确保只有在提交时才会处理
        )

    return interface
