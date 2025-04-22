import os
import json

from accelerate.test_utils.scripts.external_deps.test_ds_multiple_model import test_file_path

os.environ['HF_HOME'] = 'D:/Program of code/JetBrains/PyCharm 2025.1/running-cache'

from tokenizer_deepseek import Data_Tokenizer
from model_deepseek import ModelLoader
from train_deepseek import MathTutorTrainer
from generator_deepseek import ResponseGenerator
from datasets import Dataset

def lora_train(model_name):
    """模型训练微调方法"""
    data_tokenizer = Data_Tokenizer(model_name)
    with open("./data/dialog_deepseek.json", 'r', encoding='utf-8') as f:
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
        "./output/math_tutor_lora",
        save_embedding_layers=False,
        safe_serialization=True
    )
    with open("./output/math_tutor_lora/base_model.txt", "w") as f:
        f.write(model_name)  # 记录基础模型版本


def generate(model_name):
    """使用模型生成对话"""
    data_tokenizer = Data_Tokenizer(model_name)

    generator = ResponseGenerator(model_name, data_tokenizer.tokenizer)
    test_history = [{
        "role": "user",
        "content": "Solve the equation: 2x + 5 = 15, find the value of x."
    }]
    print(generator.generate(test_history))

    # generator = ResponseGenerator(model_name, data_tokenizer.tokenizer)
    # generator.chat_loop()

def zero_shot(model_name):
    """调用原模型批量生成回复"""
    test_file = "./data/dialog_deepseek.json"
    output_dir = "./output/zero_shot_result/"

    data_tokenizer = Data_Tokenizer(model_name)

    generator = ResponseGenerator(model_name, data_tokenizer.tokenizer)
    generator.zero_shot(test_file, output_dir)


if __name__ == "__main__":
    model_name = "./local_models/deepseek-math-7b-instruct"
    generate(model_name)

    # data_tokenizer = Data_Tokenizer(model_name)
    # with open("./data/dialog_deepseek.json", 'r', encoding='utf-8') as f:
    #     processed_data = json.load(f)
    #
    # # 转换为Dataset并分词
    # dataset = Dataset.from_dict({"messages": processed_data})
    # train_dataset = dataset.map(data_tokenizer.tokenize, batched=True,
    #                             remove_columns=["messages"], num_proc=2)