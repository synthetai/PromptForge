import sys
import os
import logging
import signal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.interface.gradio_ui import create_interface
from app.config import GRADIO_SERVER_HOST, GRADIO_SERVER_PORT
from app.model.loader import model_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def cleanup_handler(signum, frame):
    """处理程序退出时的清理工作"""
    logging.info("Cleaning up resources...")
    model_manager.cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    logging.info("Starting PromptForge application")
    iface = create_interface()
    logging.info(f"Launching Gradio interface on {GRADIO_SERVER_HOST}:{GRADIO_SERVER_PORT}")
    
    try:
        iface.launch(
            server_name=GRADIO_SERVER_HOST,
            server_port=GRADIO_SERVER_PORT,
            share=True
        )
    finally:
        model_manager.cleanup()
