from tokenizer_deepseek import Data_Tokenizer
from model_deepseek import ModelLoader
from train_deepseek import MathTutorTrainer
from generator_deepseek import ResponseGenerator
from datasets import Dataset
import json

def main():
    model_name = "deepseek-ai/DeepSeek-R1"

    data_tokenizer = Data_Tokenizer(model_name)
    with open("./data/dialog_deepseek.json", 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    # 转换为Dataset并分词
    dataset = Dataset.from_dict({"messages": processed_data})
    tokenized = dataset.map(data_tokenizer.tokenize, batched=True)

    # # 模型加载
    loader = ModelLoader(model_name)
    model = loader.load()
    #
    # # 训练
    trainer = MathTutorTrainer(model, data_tokenizer.tokenizer, tokenized)
    trainer.train()

    # 训练完成后，保存适配器，适用于LoRA等参数高效微调。
    model.save_pretrained(
        "./output/math_tutor_lora",
        save_embedding_layers=False,
        safe_serialization=True
    )
    # 同时保存基础模型信息
    with open("./output/math_tutor_lora/base_model.txt", "w") as f:
        f.write(model_name)  # 记录基础模型版本

    # 测试生成
    # generator = ResponseGenerator(model, data_tokenizer.tokenizer)
    # test_history = [{
    #     "role": "Student",
    #     "content": "Is the total cost calculation correct? My answer is $100"
    # }]
    # print(generator.generate(test_history))


if __name__ == "__main__":
    main()