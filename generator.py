from peft import PeftConfig, PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import time
from pathlib import Path
from tqdm import tqdm  # 进度条库


class ResponseGenerator:
    """使用原生模型回复生成模块"""
    GEN_CONFIG = {
        "max_new_tokens": 128, # 限制生成内容最多256个新Token(太高生成冗余内容，太低过早截断)
        "temperature": 0.3, # 控制随机性（值越低输出越稳定，值越高创意性越强）
        "top_p": 0.7, # 核采样（只保留概率累计前90%的Token，低值更集中，高值更多样）
        "top_k": 30,  # k采样，从前k个候选token采样，小k更保守（10-30），大k更开放（50-100）
        "repetition_penalty": 1.5, # 惩罚重复内容，轻度1.0~1.2，严格1.5~2.0
        "do_sample": True, # 启用采样策略，False: 贪婪解码（选准确性最高）
        "num_beams": 3,  # 1：单束，3~5：质量高但速度慢
        "early_stopping": True  # 遇到合理结果提前停止
    }

    def __init__(self, model_name, tokenizer, user_lora=False, adapter_path=""):
        # 加载模型
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 或用特定tok
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            offload_folder="./offload",  # 指定存储卸载权重的文件夹
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=True, # 生成时必须启用缓存
        )

        # 是否使用Lora适配器
        if user_lora:
            # 加载适配器配置
            self.peft_config = PeftConfig.from_pretrained(adapter_path)
            # 合并适配器，将LoRA适配器加载到基础模型
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                device_map="auto"
            )

        self.model.eval()  # 切换为评估模式
        self.gen_config = GenerationConfig(**self.GEN_CONFIG)

    def generate(self, history):
        """生成教师回复"""
        system_msg = {"role": "system", "content": "You are a mathematics tutoring assistant. Your job is to provide students with solutions to math problems."} # 系统固定提示
        full_conversation = [system_msg] + history
        # 自动设备映射
        inputs = (self.tokenizer.apply_chat_template( # 将对话转化为模型所需格式
            full_conversation,
            add_generation_prompt=True, # 在末尾添加助手标记
            return_tensors="pt", # 返回PyTorch张量
        ).to(self.model.device))

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs,  # 直接传入二维张量
                generation_config=self.gen_config,
                attention_mask=(inputs != self.tokenizer.pad_token_id)  # 动态生成注意力掩码
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

                # 显示并记录回复(带打字机效果）
                print("\n🤖 助手：", end="")
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.02)  # 调整打印速度
                print()
                history.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                print("\n\n⚠️ 用户中断，正在退出对话...")
                break

            except Exception as e:
                print(f"\n❌ 生成回复时发生错误: {str(e)}")
                print("正在重置对话历史...")
                history = []  # 重置对话以防错误累积

    def file_response(self, test_file, output_dir):
        """数据集批量生成回复"""
        # 读取测试集
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)  # 假设测试集是JSON列表格式
        # 执行批量测试
        results = []
        for idx, question in enumerate(tqdm(test_data, desc="Processing")):
            try:
                start_time = time.time()
                response = self.generate(question["messages"])
                # 记录结果
                results.append({
                    "question": question["messages"],
                    "answer": response,
                    "processing_time": time.time() - start_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                print(f"处理第 {idx + 1} 题时出错：{str(e)}")
                results.append({
                    "question": question["messages"],
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

        # 保存结果
        """保存结果到JSON文件"""
        # 创建结果目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S") # 生成时间戳字符串
        output_path = f"{output_dir}/results_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存至：{output_path}")