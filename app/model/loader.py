import torch
import gc
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor
from app.config import MODEL_CONFIGS

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.init_manager()
        return cls._instance

    def init_manager(self):
        """初始化管理器"""
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.model_type = None  # 'vl' or 'lm'

    def _clear_gpu_memory(self):
        """清理GPU显存"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
        if self.current_tokenizer is not None:
            del self.current_tokenizer
            self.current_tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        self.current_model_name = None
        self.model_type = None
        logging.info("Cleared GPU memory")

    def load_model(self, model_name, model_type, model_info):
        """通用模型加载函数"""
        try:
            # 如果要加载的模型与当前模型不同，清理显存
            if self.current_model_name != model_name or self.model_type != model_type:
                self._clear_gpu_memory()
                logging.info(f"Loading {model_type} model: {model_name}")
                
                # 获取模型配置
                model_config = MODEL_CONFIGS["default"]
                
                # 创建vllm实例
                model = LLM(
                    model=model_info["path"],  # 使用本地路径
                    trust_remote_code=True,
                    max_model_len=model_config["max_model_len"],
                    max_num_seqs=model_config["max_num_seqs"],
                    gpu_memory_utilization=model_config["gpu_memory_utilization"],
                    tensor_parallel_size=model_config["tensor_parallel_size"],
                    dtype=model_config["dtype"],
                    mm_processor_kwargs=model_config.get("mm_processor_kwargs")
                )
                
                # 加载tokenizer或processor
                if model_type == 'vl':
                    tokenizer = AutoProcessor.from_pretrained(
                        model_info["path"],
                        trust_remote_code=True
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_info["path"],
                        trust_remote_code=True
                    )
                
                self.current_model = model
                self.current_tokenizer = tokenizer
                self.current_model_name = model_name
                self.model_type = model_type
                
            return self.current_model, self.current_tokenizer
            
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def cleanup(self):
        """清理资源"""
        self._clear_gpu_memory()

# 创建全局实例
model_manager = ModelManager()
