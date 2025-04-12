from data_processor import DataProcessor
from model_loader import ModelLoader
from train_deepseek import MathTutorTrainer
from generator import ResponseGenerator
from datasets import Dataset
import json


def main():
    # 数据预处理
    data_processor = DataProcessor("deepseek-ai/DeepSeek-R1")
    with open("./data/dialog_deepseek.json", 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    # print(processed_data)

    # 转换为Dataset并分词
    dataset = Dataset.from_dict({"messages": processed_data})
    tokenized = dataset.map(data_processor.tokenize, batched=True)

    # 直接打印前3个样本的原始文本内容
    sample = tokenized[5]
    print(sample["input_ids"])
    print(sample["attention_mask"])

    sample = tokenized[205]
    print(sample["input_ids"])
    print(sample["attention_mask"])


    # # 模型加载
    # loader = ModelLoader()
    # model = loader.load()
    #
    # # 训练
    # trainer = MathTutorTrainer(model, data_processor.tokenizer, tokenized)
    # trainer.train()
    #
    # # 测试生成
    # generator = ResponseGenerator(model, data_processor.tokenizer)
    # test_history = [{
    #     "role": "Student",
    #     "content": "总成本计算正确吗？我的答案是$100"
    # }]
    # print(generator.generate(test_history))


if __name__ == "__main__":
    main()