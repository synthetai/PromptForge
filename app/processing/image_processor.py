import logging
from PIL import Image
from vllm import SamplingParams
from qwen_vl_utils import process_vision_info  # 添加这个导入
from app.model.loader import model_manager
from app.config import VL_CH_SYS_PROMPT, VL_EN_SYS_PROMPT, SAMPLING_CONFIG

def process_image(image_path, prompt, target_language, model_name):
    try:
        # 加载模型和processor
        llm, processor = model_manager.load_vl_model(model_name)
        logging.info(f"Processing image with model {model_name}: {image_path}")
        logging.info(f"Using {target_language} system prompt")
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 选择系统提示词
        system_prompt = VL_CH_SYS_PROMPT if target_language == "CH" else VL_EN_SYS_PROMPT
        
        # 构建消息
        messages = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": 28 * 28,
                    "max_pixels": 1280 * 1280,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }]
        
        # 处理提示词
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理图像输入
        image_inputs, _ = process_vision_info(messages)
        
        # 构建多模态数据
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
            
        # 构建LLM输入
        llm_inputs = {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
        }
        
        # 设置采样参数
        sampling_params = SamplingParams(**SAMPLING_CONFIG)
        
        # 生成
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        
        logging.info(f"Generated response: {response[:100]}...")
        return response.strip()
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise
