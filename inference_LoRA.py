# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
import torch


class LoRAInference:
    """LoRA适配器推理模块"""

    def __init__(self, base_model_name, adapter_path, device="cuda"):
        """
        初始化推理引擎
        :param base_model_name: 基础模型名称（如 "meta-llama/Llama-2-7b-hf"）
        :param adapter_path: LoRA适配器保存路径
        :param device: 运行设备
        """
        # 检查硬件环境
        if not torch.cuda.is_available() and device == "cuda":
            print("警告：未检测到CUDA设备，将使用CPU模式")
            device = "cpu"

        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )

        # 加载适配器配置
        self.peft_config = PeftConfig.from_pretrained(adapter_path)

        # 验证模型匹配性
        if self.base_model.config.model_type != self.peft_config.base_model_name_or_path:
            raise ValueError(
                f"基础模型类型不匹配！适配器训练于 {self.peft_config.base_model_name_or_path}，"
                f"当前加载的是 {self.base_model.config.model_type}"
            )

        # 合并适配器
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            device_map="auto" if device == "cuda" else None
        ).eval()

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 移动到指定设备
        if device != "auto":
            self.model = self.model.to(device)

        # 生成配置
        self.gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )

    def generate_response(self, prompt, history=None):
        """
        生成对话响应
        :param prompt: 当前用户输入
        :param history: 对话历史（格式：[{"role": "user", "content": "..."}, ...]）
        :return: 生成的响应文本
        """
        # 构建对话历史
        messages = [{"role": "system", "content": "你是一位专业的数学辅导教师"}]
        if history:
            messages += history
        messages.append({"role": "user", "content": prompt})

        try:
            # 编码输入
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # 生成响应
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs=inputs,
                    generation_config=self.gen_config,
                    attention_mask=(inputs != self.tokenizer.pad_token_id)
                )

            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # 后处理
            return response.strip().replace("</s>", "")

        except Exception as e:
            print(f"生成过程中发生错误：{str(e)}")
            return "抱歉，当前无法生成回答"


if __name__ == "__main__":
    # 使用示例 ================================================
    BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # 替换为实际基础模型
    ADAPTER_PATH = "./output/lora_adapters"  # 替换为实际适配器路径

    # 初始化推理引擎
    try:
        tutor = LoRAInference(
            base_model_name=BASE_MODEL,
            adapter_path=ADAPTER_PATH,
            device="cuda"  # 可改为"cpu"进行CPU推理
        )
    except Exception as e:
        print(f"初始化失败：{str(e)}")
        exit(1)

    # 模拟对话
    test_cases = [
        {"prompt": "请解释勾股定理", "history": []},
        {"prompt": "如何证明这个定理？", "history": [
            {"role": "user", "content": "请解释勾股定理"},
            {"role": "assistant", "content": "勾股定理指出在直角三角形中..."}
        ]}
    ]

    for case in test_cases:
        print(f"用户提问：{case['prompt']}")
        response = tutor.generate_response(
            prompt=case['prompt'],
            history=case['history']
        )
        print(f"教师回答：{response}\n{'-' * 40}")