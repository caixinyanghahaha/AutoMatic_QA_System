from transformers import TrainingArguments, Trainer
import torch
from utils_data import EducationDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling

model_name = "microsoft/DialoGPT-medium"
# 添加设备检测
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device}")
# 训练前再次清空缓存
if device == "mps":
    torch.mps.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型时指定torch_dtype
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "mps" else torch.float32
).to(device)
print(f"模型参数量: {model.num_parameters()/1e6:.1f}M")


def train():
    # 数据加载器优化
    train_dataset = EducationDataset("data/dialog_pairs.json", tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 不使用掩码语言模型
    )
    # 训练参数中添加设备设置
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2 if device == "mps" else 4,
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=False,  # 强制禁用FP16
        # fp16=True if torch.cuda.is_available() else False,
        bf16=True if device == "mps" else False,  # 在MPS上使用BF16
        logging_dir="./logs",
        save_steps=500,
        remove_unused_columns=False,
        optim="adamw_torch",
        report_to="none",
        gradient_accumulation_steps=2 if device == "mps" else 1  # MPS需要梯度累积
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 训练前再次清空缓存
    if device == "mps":
        torch.mps.empty_cache()

    # 启动训练
    trainer.train()

    # 保存模型
    model.save_pretrained("./model/education_dialog_model", torch_dtype=torch.float32)
    tokenizer.save_pretrained("./model/education_dialog_model")

def generate_response(model, tokenizer, context, max_length=100):
    input_text = f"{context}{tokenizer.eos_token}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # 生成回复
    output = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def test():
    # 测试示例
    context = "Tutor: 光合作用的公式是什么？"
    response = generate_response(model, tokenizer, context)
    print(f"Context: {context}\nResponse: {response}")

if __name__ == "__main__":
    test()