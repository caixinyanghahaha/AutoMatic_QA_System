import json
from torch.fx.experimental.unification.unification_tools import merge_with
from transformers import AutoTokenizer

class Data_Tokenizer:
    """数据预处理模块（内置参数配置）"""

    # 数据相关配置
    DATA_CONFIG = {
        "max_seq_length": 64,  # 适配DeepSeek的长上下文能力
        "system_prefix": "<｜begin▁of▁sentence｜>System:\n", # 系统指令前缀，用于标记来自系统的初始化提示
        "eos_token": "<｜end▁of▁sentence｜>", # 结束标记
        "thinking_prefix": ["[Step]"], # 推理引导符，指示模型生成分步思考过程（训练时自动掩码，推理时触发逐步输出）
        "add_generation_prompt": True,

        "dialog_template": (
            "<｜begin▁of▁sentence｜>You are a mathematics tutoring assistant. Your role is to guide students through Socratic questioning.{{ eos_token }}"
            "{% for message in messages[1:] %}"
                "{% if message['role'] == 'user' %}"
                    "<｜begin▁of▁sentence｜>User:\n{{ message['content'] }}{{ eos_token }}"
                "{% elif message['role'] == 'assistant' %}"
                    "<｜begin▁of▁sentence｜>Assistant:\n{{ message['content'] }}{{ eos_token }}"
                "{% endif %}"
            "{% endfor %}"
            
            "{% if add_generation_prompt %}"
                "<｜begin▁of▁sentence｜>User:\n{{ thinking_prefix }}"
            "{% endif %}"
        )
    }

    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)  # 从指定名称加载分词器
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.DATA_CONFIG["thinking_prefix"]})  # 将自定义的特殊标记添加到分词器
        # 同步配置参数
        self.DATA_CONFIG.update({
            "system_prefix": self.tokenizer.special_tokens_map.get("bos_token", ""),
            "eos_token": self.tokenizer.eos_token
        })

        self.tokenizer.chat_template = self.DATA_CONFIG["dialog_template"] # 使用预设模板
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 用结束符作为填充符

    def tokenize(self, examples):
        """分词处理，构建训练文本"""
        formatted = []
        for messages in examples["messages"]:  # messages是单个样本的消息列表
            messages = messages["messages"]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,  # 不立即分词
                    add_generation_prompt=False  # 添加生成提示符
                )
            )
        tokenized = self.tokenizer(
            formatted,  # 格式化输入数据为适合训练的文本格式
            truncation=True,  # 启用截断功能，确保不超过最大长度
            max_length=self.DATA_CONFIG["max_seq_length"],  # 指定分词之后的最大长度
            padding="max_length",  # 长度不够时将序列填充到最大长度
            padding_side="right",
            return_tensors = "pt",  # 返回PyTorch张量
            # add_special_tokens = True  # 确保添加[CLS]、[SEP]等
        )

        # 添加labels字段用于训练
        tokenized["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in seq]
            for seq in tokenized["input_ids"]
        ]

        return tokenized