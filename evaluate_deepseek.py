import json

import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import language_tool_python
from typing import List, Optional


class ResponseEvaluator:
    def __init__(self):
        # 初始化各评估组件
        self.grammar_checker = language_tool_python.LanguageTool('en-US')

        # 语义相关性模型
        self.similarity_model = pipeline(
            "text-classification",
            model="cross-encoder/stsb-roberta-base"
        )
        # 初始化情感分析管道
        self.sentiment_analyzer = pipeline("sentiment-analysis")

        # 加载通顺度模型
        self.fluency_model = AutoModelForSequenceClassification.from_pretrained("prithivida/grammar_error_correcter_v1")
        self.fluency_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")

        # 多轮对话组件
        # self.nli_model = pipeline(
        #     "text-classification",
        #     model="roberta-large-mnli",
        #     return_all_scores=True
        # )
        # 多轮对话组件
        self.nli_model = pipeline(
            # "zero-shot-classification",
            "text-classification",
            model="facebook/bart-large-mnli",
        )

    def evaluate_response(self, prompt, response, context, reference=None):
        """
        综合评估回复质量
        :param prompt: 原始提示/问题
        :param response: 生成的回复
        :param reference: 参考回复（可选）
        :return: 包含各维度评分的字典
        """
        scores = {}

        # 基础统计指标
        scores['length'] = len(response.split()) # 计算单词数量，检测回复是否过短（可能不完整）或过长（可能冗余）
        scores['unique_words'] = len(set(response.split())) / scores['length'] if scores['length'] > 0 else 0 # 唯一单词数 / 总单词数，衡量词汇多样性

        # 使用 LanguageTool 的英语语法检查器，check() 返回检测到的所有语法错误对象列表，数值越大错误越多
        scores['grammar_errors'] = len(self.grammar_checker.check(response))

        # 通顺度评分
        inputs = self.fluency_tokenizer(response, return_tensors="pt", truncation=True) # 自动截断超过模型最大长度的文本，返回 PyTorch 张量格式
        with torch.no_grad():
            outputs = self.fluency_model(**inputs)
            scores['fluency'] = float(torch.sigmoid(outputs.logits.mean()).item()) # 输出 logits 表示语法错误程度的原始分数，np.exp(-logits) 将模型输出映射到 (0,1] 区间，值越大表示越通顺（1为完美，接近0表示严重不通顺）。

        # 情感极性分析，将置信度分数转换为 [-1, 1] 区间，正向情感保持正值，负向情感转为负值，检测回复是否包含不合适的情感倾向（如负面回答礼貌问题）。
        sentiment = self.sentiment_analyzer(response)[0]
        scores['sentiment'] = sentiment['score'] * (1 if sentiment['label'] == 'POSITIVE' else -1)

        # 与问题的相关性，输出 0-1 的相似度分数（1表示完全相关）
        scores['relevance'] = float(self.similarity_model(f"{prompt} [SEP] {response}")[0]['score']) # 输入格式：问题 [SEP] 回复（[SEP] 是预定义的分隔符）

        # 增强相关性评估（考虑上下文）
        weighted_context = " ".join([
            f"[{i + 1}] {text}"
            for i, text in enumerate(context[-3:])
        ])
        scores['relevance'] = 0.7 * float(self.similarity_model(f"{prompt} [SEP] {response}")[0]['score']) + \
            0.3 * float(self.similarity_model(f"{weighted_context} [SEP] {response}")[0]['score'])

        # 与历史信息一致性检查
        # sc = []
        # for hist in context[-2:]:  # 检查最近两轮
        #     result = self.nli_model(f"{hist} [SEP] {response}", top_k=None)
        #     contradiction = next((s['score'] for s in result[0] if s['label'] == 'CONTRADICTION'), 0.0)
        #     sc.append(1 - contradiction)
        # scores['consistency'] = np.mean(sc) if scores else 1.0

        if context:
            scores_list = []
            for hist in context[-2:]:
                # 使用NLI模型检查是否矛盾
                nli_input = f"{hist} [SEP] {response}"
                result = self.nli_model(nli_input)
                # 假设第一个标签是矛盾
                contradiction_score = result[0]['score'] if result[0]['label'] == 'CONTRADICTION' else 0.0
                scores_list.append(1 - contradiction_score)

            scores['consistency'] = np.mean(scores_list) if scores_list else 1.0
        else:
            scores['consistency'] = 1.0

        # 与参考回复的相似度（如果有）
        if reference:
            scores['bleu'] = sentence_bleu([reference.split()], response.split()) # 比较生成回复与参考回复的 n-gram 重叠率
            scores['reference_similarity'] = float(self.similarity_model(f"{reference} [SEP] {response}")[0]['score']) # 输出 0-1 的相似度分数（1表示完全相关）

        return scores

    def calculate_scores(self, dataset):
        """
        评估整个数据集
        :param dataset: 包含多个对话的列表，每个对话包含question和answer字段
        :return: 包含所有对话评估结果的字典
        """
        results = {
            "dialog_scores": [],
            "overall_scores": {
                "grammar_errors": [],
                "fluency": [],
                "sentiment": [],
                "relevance": [],
                "consistency": [],
            }
        }

        for dialog in dataset:
            # 提取对话历史
            conversation = dialog["question"]
            answer = dialog["answer"]

            # 获取上下文和最后一个用户问题
            context = []
            last_prompt = ""
            for turn in conversation:
                if turn["role"] == "user":
                    last_prompt = turn["content"]
                context.append(turn["content"])

            # 评估回答
            scores = self.evaluate_response(
                prompt=last_prompt,
                response=answer,
                context=context
            )

            # 保存结果
            dialog_result = {
                "dialog_id": len(results["dialog_scores"]),
                "turn_scores": scores,
                "processing_time": dialog.get("processing_time", 0),
                "timestamp": dialog.get("timestamp", "")
            }
            results["dialog_scores"].append(dialog_result)

            # 聚合总体指标
            for key in results["overall_scores"].keys():
                results["overall_scores"][key].append(scores[key])

        # 计算总体统计量
        for key, values in results["overall_scores"].items():
            results["overall_scores"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        print(results)
        return results

# 使用示例
if __name__ == "__main__":
    evaluator = ResponseEvaluator()
    # 加载数据
    with open("./outcome/zero_shot_result/results_20250503-170711.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    evaluator.calculate_scores(raw_data)