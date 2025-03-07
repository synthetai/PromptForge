import logging
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from app.config import VL_MODELS, MODEL_CONFIGS, SAMPLING_CONFIG

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.init_manager()
        return cls._instance

    def init_manager(self):
        """初始化管理器"""
        self.loaded_models = {}
        self.loaded_processors = {}
        self.current_model_name = None

    def load_vl_model(self, model_name):
        """加载VL模型"""
        try:
            if model_name not in VL_MODELS:
                raise ValueError(f"Model {model_name} not found in configured models")

            if self.current_model_name != model_name:
                logging.info(f"Loading VL model: {model_name}")
                
                model_config = MODEL_CONFIGS["default"]
                model_info = VL_MODELS[model_name]
                
                # 创建vllm实例
                if model_name not in self.loaded_models:
                    llm = LLM(
                        model=model_info["model_path"],            # 本地模型路径
                        tokenizer=model_info["model_name"],        # HF模型名称
                        trust_remote_code=model_config["trust_remote_code"],
                        max_model_len=model_config["max_model_len"],
                        max_num_seqs=model_config["max_num_seqs"],
                        gpu_memory_utilization=model_config["gpu_memory_utilization"],
                        tensor_parallel_size=model_config["tensor_parallel_size"],
                        dtype=model_config["dtype"],
                        disable_mm_preprocessor_cache=model_config["disable_mm_preprocessor_cache"],
                        mm_processor_kwargs=model_config["mm_processor_kwargs"],
                    )
                    self.loaded_models[model_name] = llm

                # 加载processor
                if model_name not in self.loaded_processors:
                    processor = AutoProcessor.from_pretrained(
                        model_info["model_name"],
                        trust_remote_code=True
                    )
                    self.loaded_processors[model_name] = processor
                
                self.current_model_name = model_name
                
            # 返回单个对象而不是元组
            return self.loaded_models[model_name], self.loaded_processors[model_name]
            
        except Exception as e:
            logging.error(f"Error loading VL model {model_name}: {str(e)}")
            raise

    def get_available_models(self):
        """获取可用的模型列表"""
        return list(VL_MODELS.keys())

# 创建全局实例
model_manager = ModelManager()
