from transformers import GenerationConfig


class ResponseGenerator:
    """回复生成模块（内置生成参数）"""

    GEN_CONFIG = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "do_sample": True
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = GenerationConfig(**self.GEN_CONFIG)

    def generate(self, history):
        """生成教师回复"""
        inputs = self._build_inputs(history)
        outputs = self.model.generate(
            inputs,
            generation_config=self.gen_config
        )
        return self._postprocess(outputs, inputs.shape[1])

    def _build_inputs(self, history):
        """构建模型输入"""
        system_msg = {"role": "system", "content": "你是一位数学辅导教师"}
        return self.tokenizer.apply_chat_template(
            [system_msg] + history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

    def _postprocess(self, outputs, input_length):
        """后处理生成结果"""
        return self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()