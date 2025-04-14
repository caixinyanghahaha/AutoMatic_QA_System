# -*- coding: utf-8 -*-
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


class MathTutorTrainer:
    """训练模块（内置训练参数）"""

    TRAIN_CONFIG = {
        "output_dir": "./output", # 输出目录
        "per_device_train_batch_size": 2, # 每个设备的训练批次（单GPU的批次大小）
        "gradient_accumulation_steps": 4, # 梯度累积步数，用于模拟更大的批次（总批次大小= 2 * 4）
        "num_train_epochs": 15, # 训练轮次
        "learning_rate": 3e-5, # 学习率
        "logging_steps": 10, # 每10步记录一次训练日志
        "save_strategy": "epoch", # 保存策略（epoch：美伦每轮结束之后保存模型）
        "fp16": True, # 启用混合精度训练，节省显存
        "optim": "paged_adamw_32bit" # 使用分页AdamW优化器，防止显存碎片
    }

    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def train(self):
        """执行训练"""
        trainer = Trainer(
            model=self.model,
            args=self._get_training_args(), # 获取训练参数
            train_dataset=self.dataset,
            data_collator=self._get_collator() # 获取数据整理器
        )
        # trainer.train() # 启动训练
        return trainer

    def _get_training_args(self):
        """封装训练数据"""
        return TrainingArguments(
            **self.TRAIN_CONFIG,
            report_to="none", # 不向任何平台报告日志
            remove_unused_columns=False # 保留数据集中未被模型使用的列
        )

    def _get_collator(self):
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False # 禁用遮蔽语言模型（MLM）
        )