import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from config import MODEL_NAME, DEVICE

def load_model_and_processor():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor

model, processor = load_model_and_processor()
