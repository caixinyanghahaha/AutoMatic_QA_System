import json
from torch.utils.data import Dataset

class EducationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        # 拼接上下文和回复（DialoGPT格式）
        text = f"{pair['context']}{self.tokenizer.eos_token}{pair['response']}{self.tokenizer.eos_token}"
        # 编码为模型输入
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True  # 显式要求attention mask
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()  # 自回归任务中labels=input_ids
        }