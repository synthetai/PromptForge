from app.interface.gradio_ui import create_interface
from app.config import GRADIO_SERVER_HOST, GRADIO_SERVER_PORT

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(
        server_name=GRADIO_SERVER_HOST,
        server_port=GRADIO_SERVER_PORT
    )
