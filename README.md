# PromptForge

## 项目概述

**PromptForge** 是一个开源工具，旨在优化和扩展图像和视频生成的提示。用户只需输入一张图像及相应提示，PromptForge 就能利用先进的大模型技术生成更精准高效的提示。不论是图像生成还是视频生成，PromptForge 都能帮助创作者提升工作效率和生成质量。

## 功能特性

- **图片到提示扩展**：根据输入图像自动优化并生成相关提示。
- **高效易用**：只需提供图像和提示，系统将自动生成优化后的提示，简化创作流程。
- **开源可扩展**：作为开源项目，PromptForge 支持定制和二次开发，满足各种创意需求。
- **Gradio WebUI**：项目提供了基于 Gradio 开发的 WebUI，程序启动后默认监听在 0.0.0.0 的 7860 端口，方便用户通过浏览器进行交互式操作。

## 安装步骤

1. **安装依赖**

   本项目在 Ubuntu 22.04 操作系统上测试，使用的 CUDA 版本为 12.2。请在操作系统中安装以下依赖包：

   ```bash
   apt-get update && apt-get install -y git libgl1 libglib2.0-0 sox libsox-dev ffmpeg
   pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **克隆代码仓库**

   ```bash
   git clone https://github.com/yourusername/PromptForge.git
   cd PromptForge
   pip install -r requirements.txt
   ```

3. **下载模型**

   安装 Hugging Face CLI 工具并下载模型：

   ```bash
   pip install "huggingface_hub[cli]"
   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen2.5-VL-7B-Instruct
   ```

4. **修改配置文件**

   下载模型后，请更新 `app/config.py` 文件以指定本地模型路径：

   ```python
   # 模型配置
   MODEL_NAME = "/workspace/Qwen2.5-VL-7B-Instruct"
   ```

5. **启动程序**

   ```bash
   python main.py
   ```

   启动程序后，基于 Gradio 开发的 WebUI 将自动运行，并默认监听在 `0.0.0.0` 的 `7860` 端口。您可以通过浏览器访问 `http://<your-host-ip>:7860` 来使用 Web 界面。

## 联系方式

如有任何问题或建议，请联系 [wezzxn@gmail.com] 或在 GitHub 上提交 Issue。