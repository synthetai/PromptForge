import os

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# 文件路径配置
CUSTOM_TEMP_DIR = "/tmp/qwen-vl"
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)

# 设备配置
DEVICE = "cuda"  # 或 "cpu"，根据实际情况设置

# Gradio 服务器配置
GRADIO_SERVER_HOST = os.environ.get("GRADIO_SERVER_HOST", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
