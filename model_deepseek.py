from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class ModelLoader:
    """模型加载模块"""

    MODEL_CONFIG = {
        "model_name": "deepseek-ai/deepseek-math-7b",
        "quant_config": {  # 4-bit量化
            "load_in_4bit": True, # 启用4-bit量化模型，显存占用减少约75%
            "bnb_4bit_use_double_quant": True, # 对量化参数本身进行二次量化
            "bnb_4bit_quant_type": "nf4", # 使用神经网络优化的4-bit数据类型
            "bnb_4bit_compute_dtype": "bfloat16" # 指定计算时使用的数据类型，bfloat16：保持与原始模型相近的精度范围
        },
        "lora_config": {  # LoRA参数
            "r": 16, # 低秩矩阵的维度（8-64），越大训练参数越多，决定LoRA的表达能力
            "lora_alpha": 32, # 控制LoRA更新量级的缩放系数， 通常为r的两倍
            "target_modules": ["q_proj", "v_proj", "k_proj"], # 指定LoRA的目标模块，这里指定了注意力机制的核心参数
            "lora_dropout": 0.05, # Dropout，防止过拟合
            "bias": "none", # 不训练任何偏置项，lora_only：仅训练LoRA层的偏置
            "modules_to_save": ["lm_head"] # 指定需要完整训练（不冻结）的模块，lm_head（语言模型头）
        }
    }

    def load(self):
        """加载基础模型"""
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_CONFIG["model_name"],
            quantization_config=BitsAndBytesConfig(**self.MODEL_CONFIG["quant_config"]), # 载入配置文件
            device_map="auto", # 自动分割到可使用的GPU和CPU
            trust_remote_code=True # 允许执行模型自带的自定义代码
        )
        return self._apply_lora(model)

    def _apply_lora(self, model):
        """应用LoRA适配器"""
        peft_config = LoraConfig(**self.MODEL_CONFIG["lora_config"]) # 载入配置文件
        return get_peft_model(model, peft_config)