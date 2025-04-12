import json
from torch.fx.experimental.unification.unification_tools import merge_with
from transformers import AutoTokenizer


class DataProcessor:
    """数据预处理模块（内置参数配置）"""

    # 数据相关配置
    DATA_CONFIG = {
        "max_seq_length": 2048, # 序列最大长度
        "system_prompt": "You are a math tutor who needs to guide students to think by asking questions.", # 系统角色设定
        "special_tokens": ["[Tutor]", "[Student]", "[Step]"], # 自定义特殊标记
        "dialog_template": (
            # 系统提示区（带明确分隔）
            "You are a mathematics tutoring assistant. Your role is to guide students through Socratic questioning.\n\n" 
        
            # 对话历史处理
            "{% for message in messages %}"
            "{% if message['role'] == 'Student' %}"
            "Student: {{ message['content'] }}\n\n"  # 双换行分隔对话轮次
            "{% elif message['role'] == 'Tutor' %}"
            "Tutor: {{ message['content'] }}\n\n"
            "{% endif %}"
            "{% endfor %}"
            
            # 生成引导
            "Tutor: Let's think step by step. "
        ) # 对话格式化模版
    }

    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) # 从指定名称加载分词器
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.DATA_CONFIG["special_tokens"]})  # 将自定义的特殊标记添加到分词器
        self.tokenizer.chat_template = self.DATA_CONFIG["dialog_template"]

    def tokenize(self, examples):
        """分词处理"""
        return self.tokenizer( # 先前初始化时加载的HuggingFace分词器调用分词功能
            self._format_for_training(examples), # 格式化输入数据为适合训练的文本格式
            truncation=True, # 启用截断功能，确保不超过最大长度
            max_length=self.DATA_CONFIG["max_seq_length"], # 指定分词之后的最大长度
            padding="max_length", # 长度不够时将序列填充到最大长度
            # return_tensors = "pt",  # 返回PyTorch张量
            # add_special_tokens = True  # 确保添加[CLS]、[SEP]等
        )

    def _format_for_training(self, examples):
        """构建训练文本"""
        formatted = []
        for messages in examples["messages"]:  # messages是单个样本的消息列表
            messages = messages["messages"]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True, # 不立即分词
                    add_generation_prompt=True # 添加生成提示符
                )
            )
        # print(formatted)
        return formatted