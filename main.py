import os
import json
# os.environ['HF_HOME'] = 'D:/Program of code/JetBrains/PyCharm 2025.1/running-cache'

from tokenizer import Data_Tokenizer
from model import ModelLoader
from train import MathTutorTrainer
from generator import ResponseGenerator
from datasets import Dataset
import tkinter as tk
from GUI import ResponseGeneratorGUI


def lora_train(model_name):
    """模型训练微调方法"""
    data_tokenizer = Data_Tokenizer(model_name)
    with open("./data/filter_deepseek.json", 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    # 转换为Dataset并分词
    dataset = Dataset.from_dict({"messages": processed_data})
    train_dataset = dataset.map(data_tokenizer.tokenize, batched=True,
                                remove_columns=["messages"], num_proc=2)

    # 模型加载
    loader = ModelLoader(model_name)
    model = loader.load()

    # 训练
    trainer = MathTutorTrainer(model, data_tokenizer.tokenizer, train_dataset)
    trainer.train()

    # 训练完成后，保存适配器，适用于LoRA等参数高效微调。同时保存基础模型信息。
    model.save_pretrained(
        "./output/math_tutor_lora_filter_data",
        save_embedding_layers=False,
        safe_serialization=True
    )
    with open("./output/math_tutor_lora_filter_data/base_model.txt", "w") as f:
        f.write(model_name)  # 记录基础模型版本

def generate(model_name, lora=False):
    """使用模型生成对话"""
    data_tokenizer = Data_Tokenizer(model_name)
    if lora:
        adapter_path = "output/math_tutor_lora/math_tutor_lora"  # 替换为实际适配器路径
        generator = ResponseGenerator(model_name, data_tokenizer.tokenizer, True, adapter_path)
    else:
        generator = ResponseGenerator(model_name, data_tokenizer.tokenizer)
    generator.chat_loop()

def file_response(model_name, test_file, output_dir, lora=False):
    """调用原模型批量生成回复"""
    data_tokenizer = Data_Tokenizer(model_name)
    if lora:
        adapter_path = "output/math_tutor_lora_filter_data/math_tutor_lora_filter_data"  # 替换为实际适配器路径
        generator = ResponseGenerator(model_name, data_tokenizer.tokenizer, True, adapter_path)
    else:
        generator = ResponseGenerator(model_name, data_tokenizer.tokenizer)
    generator.file_response(test_file, output_dir)

def use_GUI(model_name, lora=False):
    """使用GUI交互见面进行问答"""
    data_tokenizer = Data_Tokenizer(model_name)
    if lora:
        adapter_path = "./output/math_tutor_lora"  # 替换为实际适配器路径
        generator = ResponseGenerator(model_name, data_tokenizer.tokenizer, True, adapter_path)
    else:
        generator = ResponseGenerator(model_name, data_tokenizer.tokenizer)

    root = tk.Tk()
    app = ResponseGeneratorGUI(root, generator)
    root.mainloop()

if __name__ == "__main__":
    # model_name = "./local_models/deepseek-math-7b-instruct" # 本地调用
    model_name = "deepseek-ai/deepseek-math-7b-instruct" # 远程链接
    generate(model_name,True)
    # file_response(model_name, "./data/test_deepseek.json", "./outcome/few_shot_result_filter_data/")
    # lora_train(model_name)