# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
import torch
import time


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
            device_map="auto" if device == "cuda" else None, # 自动多GPU分配
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # GPU使用bfloat16节省内存
            trust_remote_code=True # 允许模型使用自定义代码
        )

        # 加载适配器配置
        self.peft_config = PeftConfig.from_pretrained(adapter_path)

        # 验证模型匹配性
        if self.base_model.config.model_type != self.peft_config.base_model_name_or_path:
            raise ValueError( # 模型架构不匹配时报错
                f"基础模型类型不匹配！适配器训练于 {self.peft_config.base_model_name_or_path}，"
                f"当前加载的是 {self.base_model.config.model_type}"
            )

        # 合并适配器，将LoRA适配器加载到基础模型
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            device_map="auto" if device == "cuda" else None
        ).eval() # 设置为评估模式

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # 处理pad_token缺失情况
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 移动到指定设备
        if device != "auto":
            self.model = self.model.to(device) # 确保模型在目标设备上

        # 生成配置
        self.gen_config = GenerationConfig(
            max_new_tokens=512, # 限制生成最大长度
            temperature=0.7, # 控制随机性（0.0-1.0）
            top_p=0.9, # 核采样阀值
            repetition_penalty=1.2, # 重复惩罚系数
            do_sample=True # 启用采样模式
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
        # 添加当前用户输入
        messages.append({"role": "user", "content": prompt})

        try:
            # 编码输入
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True, # 添加assistant提示符
                return_tensors="pt", # 返回PyTorch张量
                truncation=True, # 启用截断
                max_length=2048 # 最大输入长度
            ).to(self.model.device) #自动对齐设备

            # 生成响应
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs=inputs,
                    generation_config=self.gen_config, # 应用生成参数
                    attention_mask=(inputs != self.tokenizer.pad_token_id) # 动态生成注意力掩码
                )

            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], # 截取新生成部分
                skip_special_tokens=True, # 跳过特殊token
                clean_up_tokenization_spaces=True # 清理空格
            )

            # 后处理
            return response.strip().replace("</s>", "") # 去除首尾空格和结束符

        except Exception as e:
            print(f"生成过程中发生错误：{str(e)}")
            return "抱歉，当前无法生成回答"


if __name__ == "__main__":
    BASE_MODEL = "deepseek-ai/DeepSeek-R1"  # 替换为实际基础模型
    ADAPTER_PATH = "./output/math_tutor_lora"  # 替换为实际适配器路径

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

    # 初始化对话历史
    conversation_history = []
    print("\n===== 数学辅导助手 =====")
    print("输入 'exit' 结束对话\n" + "=" * 30)

    while True:
        try:
            # 获取用户输入
            user_input = input("\n[你]：").strip()

            # 退出条件
            if user_input.lower() in ["exit", "quit", "退出"]:
                print("\n对话结束，再见！")
                break

            if not user_input:
                print("输入不能为空，请重新输入")
                continue

            # 生成回复
            print("\n[教师]：思考中...", end="\r")  # 显示生成状态
            response = tutor.generate_response(
                prompt=user_input,
                history=conversation_history
            )

            # 更新对话历史（保留最近6轮对话避免过长）
            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])[-12:]  # 每个对话包含user和assistant两条，保留最近6轮

            # 打印回复（带打字机效果）
            print("\n[教师]：", end="")
            for char in response:
                print(char, end="", flush=True)
                time.sleep(0.02)  # 调整打印速度
            print()

        except KeyboardInterrupt:
            print("\n检测到中断，结束对话")
            break

        except Exception as e:
            print(f"\n生成错误：{str(e)}")
            conversation_history = []  # 重置对话历史防止错误累积