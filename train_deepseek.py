# -*- coding: utf-8 -*-
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


class MathTutorTrainer:
    """训练模块（内置训练参数）"""

    TRAIN_CONFIG = {
        "output_dir": "./output",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 15,
        "learning_rate": 3e-5,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "fp16": True,
        "optim": "paged_adamw_32bit"
    }

    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def train(self):
        """执行训练"""
        trainer = Trainer(
            model=self.model,
            args=self._get_training_args(),
            train_dataset=self.dataset,
            data_collator=self._get_collator()
        )
        trainer.train()
        return trainer

    def _get_training_args(self):
        return TrainingArguments(
            **self.TRAIN_CONFIG,
            report_to="none",
            remove_unused_columns=False
        )

    def _get_collator(self):
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )