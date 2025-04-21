from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from transformers.models.esm.openfold_utils.rigid_utils import quat_multiply


class ModelLoader:
    """优化后的模型加载模块（兼容低算力GPU/CPU）"""

    MODEL_CONFIG = {
        "model_name": "",
        "quant_config": {  # 动态量化配置
            "load_in_4bit": True,  # 优先尝试4-bit量化
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
        },
        "lora_config": {
            "r": 2,
            "lora_alpha": 4,
            "target_modules": ["q_proj", "v_proj"], # 增加 k_proj, o_proj（提升效果，但增加参数）
            "lora_dropout": 0.2,
            "bias": "lora_only",
            "modules_to_save": [] # "lm_head", "embed_tokens"
        }
    }

    def __init__(self, model_name):
        self.MODEL_CONFIG["model_name"] = model_name

    def load(self):
        """加载模型（自动适配硬件环境）"""
        # 3. 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_CONFIG["model_name"],
            local_files_only=True,  # 强制本地加载
            quantization_config=self.MODEL_CONFIG["quant_config"],
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 4. 添加LoRA适配器
        peft_config = LoraConfig(
            **self.MODEL_CONFIG["lora_config"],
            task_type="CAUSAL_LM" # 指定任务为因果语言模型
        )
        model = get_peft_model(model, peft_config)

        # 训练优化设置
        model.enable_input_require_grads() # 确保输入张量的 requires_grad 属性正确传递
        # 启用梯度检查点
        # model.gradient_checkpointing_enable() # 用时间换显存，不保存全部中间激活值，反向传播时重新计算部分结果, 减少显存占用(约30-40%,速度降低20%)。

        model.config.use_cache = False  # 训练时禁用缓存

        # 打印参数量信息
        model.print_trainable_parameters()
        return model