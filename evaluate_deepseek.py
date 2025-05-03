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
        scores['length'] = len(response.split()) # 计算单词数量，检测回复是否过短（可能不完整）或过长（可能冗余）
        scores['unique_words'] = len(set(response.split())) / scores['length'] if scores['length'] > 0 else 0 # 唯一单词数 / 总单词数，衡量词汇多样性

        # 使用 LanguageTool 的英语语法检查器，check() 返回检测到的所有语法错误对象列表，数值越大错误越多
        scores['grammar_errors'] = len(self.grammar_checker.check(response))

        # 通顺度评分
        inputs = self.fluency_tokenizer(response, return_tensors="pt", truncation=True) # 自动截断超过模型最大长度的文本，返回 PyTorch 张量格式
        outputs = self.fluency_model(**inputs)
        scores['fluency'] = np.exp(-outputs.logits[0].item()) # 输出 logits 表示语法错误程度的原始分数，np.exp(-logits) 将模型输出映射到 (0,1] 区间，值越大表示越通顺（1为完美，接近0表示严重不通顺）。

        # 情感极性分析，将置信度分数转换为 [-1, 1] 区间，正向情感保持正值，负向情感转为负值，检测回复是否包含不合适的情感倾向（如负面回答礼貌问题）。
        sentiment = self.sentiment_analyzer(response)[0]
        scores['sentiment'] = sentiment['score'] * (1 if sentiment['label'] == 'POSITIVE' else -1)

        # 与问题的相关性，输出 0-1 的相似度分数（1表示完全相关）
        scores['relevance'] = float(self.similarity_model(f"{prompt} [SEP] {response}")[0]['score']) # 输入格式：问题 [SEP] 回复（[SEP] 是预定义的分隔符）

        # 与参考回复的相似度（如果有）
        if reference:
            scores['bleu'] = sentence_bleu([reference.split()], response.split()) # 比较生成回复与参考回复的 n-gram 重叠率
            scores['reference_similarity'] = float(self.similarity_model(f"{reference} [SEP] {response}")[0]['score']) # 输出 0-1 的相似度分数（1表示完全相关）

        return scores

# 使用示例
if __name__ == "__main__":
    evaluator = ResponseEvaluator()

    # 问题，回答，参考答案(可选)
    prompt = "What are the benefits of renewable energy?"
    response = "Renewable energy sources like solar and wind power can reduce greenhouse gas emissions and create jobs."
    reference = "Renewable energy offers environmental benefits through emission reduction and economic advantages via job creation."

    scores = evaluator.evaluate_response(prompt, response, reference)

    print("Response Quality Scores:")
    for k, v in scores.items():
        print(f"{k:>20}: {v:.4f}")