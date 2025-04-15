from transformers import GenerationConfig, AutoModelForCausalLM
import torch

class ResponseGenerator:
    """使用原生模型回复生成模块"""

    GEN_CONFIG = {
        "max_new_tokens": 512, # 限制生成内容最多256个新Token
        "temperature": 0.7, # 控制随机性（值越低输出越稳定）
        "top_p": 0.9, # 核采样（只保留概率累计前90%的Token）
        "top_k": 50,  # 新增top-k采样
        "repetition_penalty": 1.2, # 惩罚重复内容（大于1时抑制重复）
        "do_sample": True, # 启用采样
        "num_beams": 1,  # 显式关闭束搜索
        # "pad_token_id": 0,  # 显式指定填充token
        # "eos_token_id": 2  # 显式指定结束token
    }

    def __init__(self, model_name, tokenizer):
        # 加载模型
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model = self.model.to('cuda') if torch.cuda.is_available() else self.model
        self.gen_config = GenerationConfig(**self.GEN_CONFIG)
        # 处理可能的pad_token缺失问题
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def generate(self, history):
        """生成教师回复"""
        system_msg = {"role": "system", "content": "You are a mathematics tutoring assistant. Your role is to guide students through Socratic questioning."} # 系统固定提示
        full_conversation = [system_msg] + history
        # 自动设备映射
        inputs = (self.tokenizer.apply_chat_template( # 将对话转化为模型所需格式
            full_conversation,
            add_generation_prompt=True, # 在末尾添加助手标记
            return_tensors="pt", # 返回PyTorch张量
            truncation = True,  # 添加截断
            max_length = 2048  # 控制输入长度
        ).to(self.model.device))

        outputs = self.model.generate(
            inputs,
            generation_config=self.gen_config
        )

        return self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],  # 移除输入，只保留新生成的Token
            skip_special_tokens=True,  # 跳过特殊Token
            clean_up_tokenization_spaces = True  # 清理多余空格
        ).strip()  # 去除首尾空

    def chat_loop(self):
        """控制台多轮对话交互"""
        history = []
        print("\n=== 数学辅导对话系统 ===")
        print("输入您的问题或想法（输入 'exit' 或直接回车退出）\n")

        while True:
            try:
                # 获取用户输入
                user_input = input("👤 用户: ")

                # 退出条件判断
                if not user_input.strip() or user_input.lower() == "exit":
                    print("\n🔄 对话结束。")
                    break

                # 将用户输入加入历史
                history.append({"role": "user", "content": user_input})

                # 生成助手回复
                print("\n🤖 正在思考...", end="", flush=True)
                response = self.generate(history)
                print("\r", end="")  # 清除正在思考提示

                # 显示并记录回复
                print(f"🤖 助手: {response}\n")
                history.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                print("\n\n⚠️ 用户中断，正在退出对话...")
                break

            except Exception as e:
                print(f"\n❌ 生成回复时发生错误: {str(e)}")
                print("正在重置对话历史...")
                history = []  # 重置对话以防错误累积