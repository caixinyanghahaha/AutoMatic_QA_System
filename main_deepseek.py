from tokenizer_deepseek import Data_Tokenizer
from model_deepseek import ModelLoader
from train_deepseek import MathTutorTrainer
from generator_deepseek import ResponseGenerator
from datasets import Dataset
import json

def main():
    # 数据预处理
    model_name = "deepseek-ai/DeepSeek-R1"

    data_tokenizer = Data_Tokenizer(model_name)
    with open("./data/dialog_deepseek.json", 'r', encoding='utf-8') as f:
        processed_data = json.load(f)

    # 转换为Dataset并分词
    dataset = Dataset.from_dict({"messages": processed_data})
    tokenized = dataset.map(data_tokenizer.tokenize, batched=True)

    # 模型加载
    loader = ModelLoader()
    model = loader.load()

    # 训练
    trainer = MathTutorTrainer(model, data_tokenizer.tokenizer, tokenized)
    trainer.train()

    # 测试生成
    # generator = ResponseGenerator(model, data_tokenizer.tokenizer)
    # test_history = [{
    #     "role": "Student",
    #     "content": "总成本计算正确吗？我的答案是$100"
    # }]
    # print(generator.generate(test_history))


if __name__ == "__main__":
    main()