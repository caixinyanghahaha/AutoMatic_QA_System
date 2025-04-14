from transformers import GenerationConfig


class ResponseGenerator:
    """回复生成模块"""

    GEN_CONFIG = {
        "max_new_tokens": 256, # 限制生成内容最多256个新Token
        "temperature": 0.7, # 控制随机性（值越低输出越稳定）
        "top_p": 0.9, # 核采样（只保留概率累计前90%的Token）
        "repetition_penalty": 1.2, # 惩罚重复内容（大于1时抑制重复）
        "do_sample": True # 启用采样
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = GenerationConfig(**self.GEN_CONFIG)

    def generate(self, history):
        """生成教师回复"""
        system_msg = {"role": "system", "content": "你是一位数学辅导教师"} # 系统固定提示
        full_conversation = [system_msg] + history

        inputs = self.tokenizer.apply_chat_template( # 将对话转化为模型所需格式
            full_conversation,
            add_generation_prompt=True, # 在末尾添加助手标记
            return_tensors="pt" # 返回PyTorch张量
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs,
            generation_config=self.gen_config
        )

        return self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],  # 移除输入，只保留新生成的Token
            skip_special_tokens=True  # 跳过特殊Token
        ).strip()  # 去除首尾空