import logging
from qwen_vl_utils import process_vision_info
from app.model.loader import model, processor

VL_CH_SYS_PROMPT = '''你是一位Prompt优化师，旨在参考用户输入的图像的细节内容，把用户输入的Prompt改写为优质Prompt，使其更完整、更具表现力，同时不改变原意。你需要综合用户输入的照片内容和输入的Prompt进行改写，严格参考示例的格式进行改写。
任务要求：
1. 对于过于简短的用户输入，在不改变原意前提下，合理推断并补充细节，使得画面更加完整好看；
2. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等）、画面风格、空间关系、镜头景别；
3. 整体中文输出，保留引号、书名号中原文以及重要的输入信息，不要改写；
4. Prompt应匹配符合用户意图且精准细分的风格描述。如果用户未指定，则根据用户提供的照片的风格，你需要仔细分析照片的风格，并参考风格进行改写；
5. 如果Prompt是古诗词，应该在生成的Prompt中强调中国古典元素，避免出现西方、现代、外国场景；
6. 你需要强调输入中的运动信息和不同的镜头运镜；
7. 你的输出应当带有自然运动属性，需要根据描述主体目标类别增加这个目标的自然动作，描述尽可能用简单直接的动词；
8. 你需要尽可能的参考图片的细节信息，如人物动作、服装、背景等，强调照片的细节元素；
9. 改写后的prompt字数控制在80-100字左右
10. 无论用户输入什么语言，你都必须输出中文
直接输出改写后的文本。'''

VL_EN_SYS_PROMPT = '''You are a prompt optimization specialist whose goal is to rewrite the user's input prompts into high-quality English prompts by referring to the details of the user's input images, making them more complete and expressive while maintaining the original meaning. You need to integrate the content of the user's photo with the input prompt for the rewrite, strictly adhering to the formatting of the examples provided.
Task Requirements:
1. For overly brief user inputs, reasonably infer and supplement details without changing the original meaning, making the image more complete and visually appealing;
2. Improve the characteristics of the main subject in the user's description (such as appearance, expression, quantity, ethnicity, posture, etc.), rendering style, spatial relationships, and camera angles;
3. The overall output should be in Chinese, retaining original text in quotes and book titles as well as important input information without rewriting them;
4. The prompt should match the user's intent and provide a precise and detailed style description. If the user has not specified a style, you need to carefully analyze the style of the user's provided photo and use that as a reference for rewriting;
5. If the prompt is an ancient poem, classical Chinese elements should be emphasized in the generated prompt, avoiding references to Western, modern, or foreign scenes;
6. You need to emphasize movement information in the input and different camera angles;
7. Your output should convey natural movement attributes, incorporating natural actions related to the described subject category, using simple and direct verbs as much as possible;
8. You should reference the detailed information in the image, such as character actions, clothing, backgrounds, and emphasize the details in the photo;
9. Control the rewritten prompt to around 80-100 words.
10. No matter what language the user inputs, you must always output in English.
Directly output the rewritten English text.'''

def process_image(image_path, prompt, target_language):
    system_prompt = VL_CH_SYS_PROMPT if target_language == "CH" else VL_EN_SYS_PROMPT

    logging.info(f"Processing image: {image_path}")
    logging.info(f"User prompt: {prompt}")
    logging.info(f"Target language: {target_language}")
    logging.info(f"System prompt used: {system_prompt}")

    messages = [{
        'role': 'system',
        'content': system_prompt
    }, {
        'role': 'user',
        'content': [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    logging.info(f"Generated output: {output_text[0][:100]}...")  # 只记录输出的前100个字符

    return output_text[0]
