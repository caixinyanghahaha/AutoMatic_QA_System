from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import json
import time
from pathlib import Path
from tqdm import tqdm  # 进度条库

class ZeroShotGenerator:

    GEN_CONFIG = {
        "max_new_tokens": 512,  # 限制生成内容最多256个新Token
        "temperature": 0.7,  # 控制随机性（值越低输出越稳定）
        "top_p": 0.9,  # 核采样（只保留概率累计前90%的Token）
        "top_k": 50,  # 新增top-k采样
        "repetition_penalty": 1.2,  # 惩罚重复内容（大于1时抑制重复）
        "do_sample": True,  # 启用采样
        "num_beams": 1,  # 显式关闭束搜索
        # "pad_token_id": 0,  # 显式指定填充token
        # "eos_token_id": 2  # 显式指定结束token
    }

    def __init__(self, model_name, tokenizer):

        # 初始化配置
        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def test_zero_shot(self, test_file, output_dir):
        # 读取测试集
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)  # 假设测试集是JSON列表格式

        # 执行批量测试
        results = []

        for idx, question in enumerate(tqdm(test_data, desc="Processing")):
            try:
                start_time = time.time()

                system_msg = {"role": "system",
                              "content": "You are a mathematics tutoring assistant. Your role is to guide students through Socratic questioning."}  # 系统固定提示
                full_conversation = [system_msg] + question
                # 自动设备映射
                inputs = self.tokenizer.apply_chat_template(  # 将对话转化为模型所需格式
                    full_conversation,
                    add_generation_prompt=True,  # 在末尾添加助手标记
                    return_tensors="pt",  # 返回PyTorch张量
                    truncation=True,  # 添加截断
                    max_length=2048  # 控制输入长度
                ).to(self.model.device)

                # 生成回答
                outputs = self.model.generate(**inputs, **self.gen_config)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces = True)
                clean_response = response[len(system_msg):].strip()

                # 记录结果
                results.append({
                    "question": question,
                    "answer": clean_response,
                    "processing_time": time.time() - start_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                print(f"处理第 {idx + 1} 题时出错：{str(e)}")
                results.append({
                    "question": question,
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