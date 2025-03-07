import gradio as gr
from app.processing.image_processor import process_image
from app.processing.text_processor import process_text
from app.config import DEFAULT_VL_MODEL, VL_MODELS, LM_MODELS

def create_interface():
    def handle_image_and_prompt(image, prompt, target_language, model_name=DEFAULT_VL_MODEL):
        if image is None:
            gr.Warning("Please upload an image first!")
            return "Please upload an image."
        try:
            gr.Info(f"Processing with model: {model_name}")
            enhanced_prompt = process_image(image, prompt, target_language, model_name)
            gr.Info("Processing completed!")
            return enhanced_prompt
        except Exception as e:
            gr.Error(f"Error: {str(e)}")
            return f"Error occurred: {str(e)}"

    def handle_text_prompt(prompt, target_language, model_name):
        if not prompt.strip():
            gr.Warning("Please enter a prompt!")
            return "Please enter a prompt."
        try:
            gr.Info(f"Processing with model: {model_name}")
            enhanced_prompt = process_text(prompt, target_language, model_name)
            gr.Info("Processing completed!")
            return enhanced_prompt
        except Exception as e:
            gr.Error(f"Error: {str(e)}")
            return f"Error occurred: {str(e)}"

    with gr.Blocks(title="PromptForge") as interface:
        gr.Markdown("# PromptForge")

        with gr.Tabs() as tabs:
            # I2V Prompt 标签页
            with gr.Tab("I2V Prompt") as i2v_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="filepath",
                            label="Upload Image",
                            height=400
                        )
                        prompt_input = gr.Textbox(
                            label="Input Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3
                        )
                        with gr.Row():
                            language_input = gr.Radio(
                                choices=["CH", "EN"],
                                value="CH",
                                label="Target Language"
                            )
                            model_select = gr.Dropdown(
                                choices=list(VL_MODELS.keys()),
                                value=DEFAULT_VL_MODEL,
                                label="Select Model"
                            )
                        submit_btn = gr.Button(
                            "Generate Enhanced Prompt"
                        )
                    
                    with gr.Column(scale=1):
                        output = gr.Textbox(
                            label="Enhanced Prompt Output",
                            lines=12,
                            show_copy_button=True
                        )

                submit_btn.click(
                    fn=handle_image_and_prompt,
                    inputs=[
                        image_input,
                        prompt_input,
                        language_input,
                        model_select
                    ],
                    outputs=output
                )
            # T2I Prompt 标签页
            with gr.Tab("T2I Prompt") as t2i_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        text_prompt_input = gr.Textbox(
                            label="Input Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3
                        )
                        with gr.Row():
                            text_language_input = gr.Radio(
                                choices=["CH", "EN"],
                                value="CH",
                                label="Target Language"
                            )
                            text_model_select = gr.Dropdown(
                                choices=list(LM_MODELS.keys()),
                                value="Qwen2.5-7B-Instruct",
                                label="Select Model"
                            )
                        text_submit_btn = gr.Button(
                            "Generate Enhanced Prompt"
                        )
                    
                    with gr.Column(scale=1):
                        text_output = gr.Textbox(
                            label="Enhanced Prompt Output",
                            lines=12,
                            show_copy_button=True
                        )

                text_submit_btn.click(
                    fn=handle_text_prompt,
                    inputs=[
                        text_prompt_input,
                        text_language_input,
                        text_model_select
                    ],
                    outputs=text_output
                )

    return interface
