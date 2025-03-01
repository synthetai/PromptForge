# PromptForge

## Project Overview

**PromptForge** is an open-source tool designed to optimize and extend prompts for image and video generation. Users simply input an image and a related prompt, and PromptForge utilizes advanced large-model techniques to optimize these prompts, generating more accurate and efficient outputs. Whether for image generation or video generation, PromptForge helps creators improve both efficiency and quality.

## Features

- **Image-to-Prompt Extension**: Automatically optimize and generate relevant prompts based on the input image.
- **Efficient & Easy-to-Use**: Just input an image and a prompt, and the system will automatically generate optimized prompts, streamlining the creative process.
- **Open-Source and Extensible**: As an open-source project, PromptForge supports customization and further development, making it suitable for various creative needs.

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/PromptForge.git
cd PromptForge
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Additional Installation Steps

1. **Install System Dependencies**  
   The project was tested on Ubuntu 22.04 with CUDA version 12.2. Install the following packages on your operating system:
   ```bash
   apt-get update && apt-get install -y git libgl1 libglib2.0-0 sox libsox-dev ffmpeg
   pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **Download the Model**  
   Install the Hugging Face CLI tool and download the model:
   ```bash
   pip install "huggingface_hub[cli]"
   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen2.5-VL-7B-Instruct
   ```

3. **Modify the Configuration File**  
   After downloading the model, update the `config.py` file to specify the local model path:
   ```python
   # Model Configuration
   MODEL_NAME = "/workspace/Qwen2.5-VL-7B-Instruct"
   ```

4. **Start the Program**  
   
   ```bash
   python main.py
   ```

## Contact

For any questions or suggestions, please contact [wezzxn@gmail.com] or raise an Issue on GitHub.