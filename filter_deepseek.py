"""
英文对话数据筛选系统 v1.2
功能：实现数据清洗 -> 多样性采样 -> 信息量评估 -> 人工验证 全流程
环境要求：Python 3.8+，依赖见requirements.txt
作者：AI助手
"""
import json
# -*- coding: utf-8 -*-
import re
import numpy as np
# import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
from language_tool_python import LanguageTool
from sklearn.model_selection import StratifiedShuffleSplit
from textblob import TextBlob
from tqdm import tqdm


class DialogueFilter:
    FILTER_CONFIG = {
        "input_path": "dialogues.json",  # 输入数据路径
        "output_path": "selected_top20.json",  # 输出路径
        "target_size": 20,  # 最终保存的对话数量
        "min_turns": 3,  # 过滤少于三轮对话
        "min_length": 15,  # 每轮最少词数
        "grammar_error_threshold": 0.1,  # 每个词最多0.1个语法错误
        "intent_labels": ["request", "complaint", "query", "social", "instruction"],  # 意图分类标签（request：服务请求；complaint：投诉；query：信息查询；social：社交对话；instruction：操作指导）
        "device": "auto",
        "model": "joeddav/xlm-roberta-large-xnli", # XLM-RoBERTa大型模型（跨语言版，支持多语言零样本分类无需微调）
    }

    def __init__(self, config=None):
        # 初始化工具和模型
        self.grammar_tool = LanguageTool('en-US')
        self.intent_classifier = self._init_intent_classifier()
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.uncertainty_model, self.tokenizer = self._init_uncertainty_model()

    def preprocess(self,data):
        """
        数据清洗与基础过滤
        返回：过滤后的对话列表
        """
        filtered = []
        tool = LanguageTool('en-US')

        def contains_private_info(dialog):
            """使用正则表达式检测隐私信息"""
            patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # 社会安全号码格式，SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
            ]
            for turn in dialog['turns']:
                text = turn['text']
                for pattern in patterns:
                    if re.search(pattern, text):
                        return True
            return False

        for dialog in tqdm(data, desc="Preprocessing"):
            # 对话轮次不足直接跳过
            if len(dialog['turns']) < self.FILTER_CONFIG["min_turns"]:
                continue

            # 检查每轮长度，任意轮不满足最小长度即标记为无效
            valid = True
            for turn in dialog['turns']:
                words = turn['text'].split()
                if len(words) < self.FILTER_CONFIG["min_length"]:
                    valid = False
                    break

                # 先进行拼写纠正再检查语法
                corrected = str(TextBlob(turn['text']).correct())  # 纠正拼写

                # 计算语法错误密度，错误数 / 总词数 > 阀值则过滤
                matches = tool.check(corrected)
                error_density = len(matches) / len(words)
                if error_density > self.FILTER_CONFIG["grammar_error_threshold"]:
                    valid = False
                    break

            # 检查隐私信息
            if valid and not contains_private_info(dialog):
                filtered.append(dialog)

        return filtered

    def diversity_sampling(self,filtered_data):
        """
        基于意图和结构的多样性采样
        返回：聚类后的代表性样本
        """
        # 意图分类
        intent_classifier = pipeline(
            "zero-shot-classification",
            model=self.FILTER_CONFIG["model"],
            device=self.FILTER_CONFIG["device"],
        )
        # 提取每个对话第一轮的意图，并以此作为判断依据
        intents = []
        for dialog in tqdm(filtered_data, desc="Intent Analysis"):
            first_turn = dialog['turns'][0]['text'] # 获取首轮输入
            result = intent_classifier( # 使用分类器预测意图标签
                first_turn,
                candidate_labels=self.FILTER_CONFIG["intent_labels"] # 指定候选标签集
            )
            intents.append(result['labels'][0]) # 选择置信度最高的标签插入

        # 分层抽样保证意图分布，使得60个样本的分布与原始数据相同
        sss = StratifiedShuffleSplit( # 创建分层随机抽样对象
            n_splits=1, # 数据分割次数
            test_size=60,
            random_state=42 # 固定随机种子确保可复现性
        )
        _, idxs = next(sss.split(filtered_data, intents)) # 根据 intents 标签分层，idxs存储测试集的索引
        stratified_selected = [filtered_data[i] for i in idxs] # 保存分层抽样结果

        # 基于对话结构的K-means聚类，提取结构特征：话轮数、平均长度、问答比
        features = []
        for dialog in stratified_selected:
            turns = len(dialog['turns']) # 计算总轮次
            avg_length = np.mean([len(t['text'].split()) for t in dialog['turns']]) # 计算平均对话长度
            q_ratio = sum(1 for t in dialog['turns'] if t['role'] == 'user') / turns # 生成器表达式，计算每轮文本的单词数，并计算用户轮次占比
            features.append([turns, avg_length, q_ratio]) # 对话总轮次，平均对话数，用户话论占比，用于后续的聚类分析
        # 聚类
        kmeans = KMeans(n_clusters=15) # 将 60 个样本划分为 15 个簇
        clusters = kmeans.fit_predict(features) # 返回每个样本的簇标签
        # 选择代表样本
        final_selected = []
        for cluster_id in range(15):
            cluster_data = [d for d, c in zip(final_selected, clusters) if c == cluster_id]
            # 从每个簇中随机选1个代表
            if cluster_data:
                final_selected.append(np.random.choice(cluster_data))
        return final_selected

    def information_ranking(self,data):
        """
        综合评估对话的信息密度和训练价值
        返回：按价值排序的对话列表
        """
        # TF-IDF信息密度
        tfidf = TfidfVectorizer(stop_words='english') # 用于计算词频-逆文档频率，忽略英文停用词
        texts = [" ".join(t['text'] for t in d['turns']) for d in data] # 将对话中的每个轮次的文本连接成一个字符串
        tfidf_scores = tfidf.fit_transform(texts).sum(axis=1).A1 # 使用fit_transform方法将文本转换为TF-IDF特征矩阵，并计算每个文本的TF-IDF总分（按行求和），并转化为一维数组。

        # 语义密度（BERT编码）使用Sentence-BERT计算语义密度
        model = SentenceTransformer('all-mpnet-base-v2') # 用于生成橘子嵌入
        embeddings = model.encode(texts) # 将文本输入模型生成嵌入向量
        semantic_scores = np.linalg.norm(embeddings, axis=1) # 计算每个嵌入向量的长度，作为语义密度的评分。

        # 基于BERT的不确定性采样
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(self.FILTER_CONFIG["device"])
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        uncertainties = [] # 用于存储每个对话的熵（不确定性评分）
        for dialog in tqdm(data, desc="Uncertainty Scoring"):
            inputs = tokenizer(
                dialog['turns'][-1]['text'],
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.FILTER_CONFIG["device"])
            with torch.no_grad():
                outputs = model(**inputs).logits # 获取模型的输出，即分类的未归一化分数
            prob = torch.softmax(outputs, dim=1)[0] # 计算softmax值，将logits转化为概率分布
            entropy = -torch.sum(prob * torch.log(prob)).item() # 计算熵，表示不确定性评分。熵越高，表示模型对该输入的预测越不确定
            uncertainties.append(entropy) # 将计算得到的熵添加到不确定性列表中
        uncertainty_scores = np.array(uncertainties) # 将不确定性列表转换为NumPy数组

        # 综合评分
        combined = (
                0.4 * tfidf_scores +
                0.3 * semantic_scores +
                0.3 * uncertainty_scores
        )
        return [data[i] for i in np.argsort(combined)[-self.FILTER_CONFIG["target_size"]:]]


    def tu_xiaoniu(self):
        # 加载数据
        input_file = "asssaad"
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 阶段1：预处理
        cleaned = self.preprocess(raw_data)
        print(f"预处理后剩余对话：{len(cleaned)}条")

        # 阶段2：多样性采样
        diverse = self.diversity_sampling(cleaned)

        # 阶段3：信息量评估
        ranked = self.information_ranking(diverse)

        # 输出Top候选
        print("\n推荐的候选对话：")
        for i, dialog in enumerate(ranked[:20]):
            print(f"{i + 1}. ID:{dialog['id']} 话轮数:{len(dialog['turns'])}")