# -*- coding: utf-8 -*-
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch

torch.backends.cudnn.benchmark = True         # 自动优化算法

class MathTutorTrainer:
    """训练模块"""
    TRAIN_CONFIG = {
        "save_strategy": "no",  # 关闭保存策略
        "save_steps": None,  # 必须设为None
        "save_total_limit": None,  # 关闭检查点限制
        "no_cuda": False,  # 保持GPU加速
        "label_names": "labels",  # 明确指定标签字段
        "output_dir": "./output", # 输出目录
        "per_device_train_batch_size": 1, # 每个设备的训练批次（单GPU的批次大小）
        "gradient_accumulation_steps": 8, # 梯度累积步数，用于模拟更大的批次（总批次大小= 2 * 4）
        "num_train_epochs": 30, # 训练轮次
        "learning_rate": 3e-5, # 学习率
        "logging_steps": 50, # 每50步记录一次训练日志
        "fp16": torch.cuda.get_device_capability()[0] >= 7,  # 根据GPU架构自动启用, Volta架构及以上启用
        "optim": "adamw_bnb_8bit", # 使用分页AdamW优化器，防止显存碎片
        "dataloader_num_workers": 2,  # 提升数据加载并行度
        "dataloader_pin_memory": True,  # 锁页内存加速传输
        "gradient_checkpointing": True,  # 启用梯度检查点, 节省30-40%显存
        "eval_steps": 50,  # 添加验证步骤
        "report_to": "tensorboard",  # 启用TensorBoard监控, 实时监控GPU利用率
        "logging_dir": "./logs",
        "remove_unused_columns": False,
    }

    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.model.train()  # 显式设置为训练模式
        self.tokenizer = tokenizer
        self.dataset = dataset

        # GPU内存优化
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.model = self.model.to('cuda')

    def train(self):
        """执行GPU优化训练"""
        # 混合精度配置
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                **self.TRAIN_CONFIG,
            ),  # 获取训练参数
            train_dataset=self.dataset,
            data_collator=DataCollatorForLanguageModeling( # GPU优化的数据整理器
                tokenizer=self.tokenizer,
                mlm=False,  # 禁用遮蔽语言模型（MLM）
                pad_to_multiple_of=8  # 对齐显存访问
            )  # 获取数据整理器
            # eval_dataset=self.dataset.select(range(100))  # 示例验证集
        )

        # 显存占用分析
        print("初始显存占用:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        print(f"当前占用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        trainer.train()

        return trainer