import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.interface.gradio_ui import create_interface
from app.config import GRADIO_SERVER_HOST, GRADIO_SERVER_PORT

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    logging.info("Starting PromptForge application")
    iface = create_interface()
    logging.info(f"Launching Gradio interface on {GRADIO_SERVER_HOST}:{GRADIO_SERVER_PORT}")
    iface.launch(
        server_name=GRADIO_SERVER_HOST,
        server_port=GRADIO_SERVER_PORT
    )
