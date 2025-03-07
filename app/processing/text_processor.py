import logging
from vllm import SamplingParams
from app.model.loader import model_manager
from app.config import LM_MODELS, LM_ZH_SYS_PROMPT, LM_EN_SYS_PROMPT

def process_text(prompt, target_language, model_name):
    """处理纯文本提示词"""
    try:
        # 通过ModelManager加载模型和tokenizer
        llm, tokenizer = model_manager.load_model(
            model_name=model_name,
            model_type='lm',
            model_info=LM_MODELS[model_name]  # 传递模型信息字典
        )
        
        logging.info(f"Processing text with model {model_name}")
        
        # 选择系统提示词
        system_prompt = LM_ZH_SYS_PROMPT if target_language == "CH" else LM_EN_SYS_PROMPT
        
        # 准备消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=512
        )
        
        # 生成输出
        outputs = llm.generate([text], sampling_params)
        response = outputs[0].outputs[0].text
        
        logging.info(f"Generated response: {response[:100]}...")
        return response.strip()
        
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        raise
