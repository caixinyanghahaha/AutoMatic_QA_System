import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import language_tool_python


class ResponseEvaluator:
    def __init__(self):
        # 初始化各评估组件
        self.grammar_checker = language_tool_python.LanguageTool('en-US')

        # 语义相关性模型（需要提前下载模型）
        self.similarity_model = pipeline(
            "text-classification",
            model="cross-encoder/stsb-roberta-base"
        )

        # 初始化情感分析管道
        self.sentiment_analyzer = pipeline("sentiment-analysis")

        # 加载通顺度模型（需要提前下载模型）
        self.fluency_model = AutoModelForSequenceClassification.from_pretrained("prithivida/grammar_error_correcter_v1")
        self.fluency_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")

    def evaluate_response(self, prompt, response, reference=None):
        """
        综合评估回复质量
        :param prompt: 原始提示/问题
        :param response: 生成的回复
        :param reference: 参考回复（可选）
        :return: 包含各维度评分的字典
        """
        scores = {}

        # 基础统计指标
        scores['length'] = len(response.split())
        scores['unique_words'] = len(set(response.split())) / scores['length'] if scores['length'] > 0 else 0

        # 语法检查
        scores['grammar_errors'] = len(self.grammar_checker.check(response))

        # 通顺度评分
        scores['fluency'] = self._calculate_fluency(response)

        # 情感极性分析
        sentiment = self.sentiment_analyzer(response)[0]
        scores['sentiment'] = sentiment['score'] * (1 if sentiment['label'] == 'POSITIVE' else -1)

        # 与问题的相关性
        scores['relevance'] = self._calculate_similarity(prompt, response)

        # 与参考回复的相似度（如果有）
        if reference:
            scores['bleu'] = sentence_bleu([reference.split()], response.split())
            scores['reference_similarity'] = self._calculate_similarity(reference, response)

        return scores

    def _calculate_fluency(self, text):
        """使用预训练模型计算通顺度"""
        inputs = self.fluency_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.fluency_model(**inputs)
        return np.exp(-outputs.logits[0].item())

    def _calculate_similarity(self, text1, text2):
        """计算语义相似度"""
        return float(self.similarity_model(f"{text1} [SEP] {text2}")[0]['score'])


# 使用示例
if __name__ == "__main__":
    evaluator = ResponseEvaluator()

    prompt = "What are the benefits of renewable energy?"
    response = "Renewable energy sources like solar and wind power can reduce greenhouse gas emissions and create jobs."
    reference = "Renewable energy offers environmental benefits through emission reduction and economic advantages via job creation."

    scores = evaluator.evaluate_response(prompt, response, reference)

    print("Response Quality Scores:")
    for k, v in scores.items():
        print(f"{k:>20}: {v:.4f}")