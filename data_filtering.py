"""
英文对话数据筛选系统 v1.2
功能：实现数据清洗 -> 多样性采样 -> 信息量评估 -> 人工验证 全流程
环境要求：Python 3.8+，依赖见requirements.txt
作者：AI助手
"""

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
# from convokit import Corpus, Conversation, ConversationParser
from tqdm import tqdm


# ------------------
# 配置区（根据需求修改）
# ------------------
class Config:
    input_path = "dialogues.json"  # 输入数据路径
    output_path = "selected_top20.json"  # 输出路径
    target_size = 20  # 目标筛选数量
    min_turns = 3  # 对话最少轮次
    min_length = 15  # 每轮最少词数
    grammar_error_threshold = 0.1  # 语法错误密度阈值
    intent_labels = ["request", "complaint", "query", "social", "instruction"]  # 意图分类标签
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备


# ------------------
# 数据加载（示例）
# ------------------
def load_data(file_path):
    """
    模拟加载对话数据，实际应替换为真实数据加载
    数据结构示例：
    [{
        "id": "dialog_001",
        "turns": [
            {"role": "user", "text": "I can't log in to my account..."},
            {"role": "agent", "text": "Have you tried resetting..."}
        ]
    }]
    """
    # 示例数据（实际使用时替换为真实数据加载）
    return [
        {
            "id": f"dialog_{i:03}",
            "turns": [
                {"role": "user", "text": "Sample user utterance"},
                {"role": "agent", "text": "Sample agent response"}
            ]
        } for i in range(300)
    ]


# ------------------
# 阶段1：数据预处理
# ------------------
def preprocess(data):
    """
    数据清洗与基础过滤
    返回：过滤后的对话列表
    """
    filtered = []
    tool = LanguageTool('en-US')

    for dialog in tqdm(data, desc="Preprocessing"):
        # 过滤短对话
        if len(dialog['turns']) < Config.min_turns:
            continue

        # 检查每轮长度
        valid = True
        for turn in dialog['turns']:
            words = turn['text'].split()
            if len(words) < Config.min_length:
                valid = False
                break

            # 语法检查
            matches = tool.check(turn['text'])
            error_density = len(matches) / len(words)
            if error_density > Config.grammar_error_threshold:
                valid = False
                break

        # 隐私信息过滤
        if valid and not contains_private_info(dialog):
            filtered.append(dialog)

    return filtered


def contains_private_info(dialog):
    """使用正则表达式检测隐私信息"""
    patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
    ]
    for turn in dialog['turns']:
        text = turn['text']
        for pattern in patterns:
            if re.search(pattern, text):
                return True
    return False


# ------------------
# 阶段2：多样性采样
# ------------------
def diversity_sampling(filtered_data):
    """
    基于意图和结构的多样性采样
    返回：聚类后的代表性样本
    """
    # 意图分类
    intent_classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        device=Config.device
    )

    # 提取意图特征
    intents = []
    for dialog in tqdm(filtered_data, desc="Intent Analysis"):
        first_turn = dialog['turns'][0]['text']
        result = intent_classifier(
            first_turn,
            candidate_labels=Config.intent_labels
        )
        intents.append(result['labels'][0])

    # 分层抽样保证意图分布
    selected = stratified_sampling(filtered_data, intents, n_samples=60)

    # 对话结构聚类
    return structural_clustering(selected, n_clusters=15)


def stratified_sampling(data, labels, n_samples):
    """分层随机抽样"""
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=n_samples,
        random_state=42
    )
    _, idxs = next(sss.split(data, labels))
    return [data[i] for i in idxs]


def structural_clustering(data, n_clusters):
    """基于对话结构的K-means聚类"""
    # 提取结构特征：话轮数、平均长度、问答比
    features = []
    for dialog in data:
        turns = len(dialog['turns'])
        avg_length = np.mean([len(t['text'].split()) for t in dialog['turns']])
        q_ratio = sum(1 for t in dialog['turns'] if t['role'] == 'user') / turns
        features.append([turns, avg_length, q_ratio])

    # 聚类并选择代表样本
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(features)

    selected = []
    for cluster_id in range(n_clusters):
        cluster_data = [d for d, c in zip(data, clusters) if c == cluster_id]
        # 从每个簇中随机选1个代表
        if cluster_data:
            selected.append(np.random.choice(cluster_data))

    return selected


# ------------------
# 阶段3：信息量评估
# ------------------
def information_ranking(data):
    """
    综合评估对话的信息密度和训练价值
    返回：按价值排序的对话列表
    """
    # TF-IDF信息密度
    tfidf = TfidfVectorizer(stop_words='english')
    texts = [" ".join(t['text'] for t in d['turns']) for d in data]
    tfidf_scores = tfidf.fit_transform(texts).sum(axis=1).A1

    # 语义密度（BERT编码）
    semantic_scores = get_semantic_density(texts)

    # 不确定性采样
    uncertainty_scores = active_learning_scores(data)

    # 综合评分
    combined = (
            0.4 * tfidf_scores +
            0.3 * semantic_scores +
            0.3 * uncertainty_scores
    )
    return [data[i] for i in np.argsort(combined)[-Config.target_size * 2:]]


def get_semantic_density(texts):
    """使用Sentence-BERT计算语义密度"""
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts)
    return np.linalg.norm(embeddings, axis=1)


def active_learning_scores(data):
    """基于BERT的不确定性采样"""
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(Config.device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    uncertainties = []
    for dialog in tqdm(data, desc="Uncertainty Scoring"):
        inputs = tokenizer(
            dialog['turns'][-1]['text'],
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(Config.device)

        with torch.no_grad():
            outputs = model(**inputs).logits

        prob = torch.softmax(outputs, dim=1)[0]
        entropy = -torch.sum(prob * torch.log(prob)).item()
        uncertainties.append(entropy)

    return np.array(uncertainties)


# ------------------
# 主流程
# ------------------
if __name__ == "__main__":
    # 加载数据
    raw_data = load_data(Config.input_path)

    # 阶段1：预处理
    cleaned = preprocess(raw_data)
    print(f"预处理后剩余对话：{len(cleaned)}条")

    # 阶段2：多样性采样
    diverse = diversity_sampling(cleaned)

    # 阶段3：信息量评估
    ranked = information_ranking(diverse)

    # 输出Top候选（需人工验证）
    print("\n推荐进行人工评估的候选对话（前40条）：")
    for i, dialog in enumerate(ranked[:40]):
        print(f"{i + 1}. ID:{dialog['id']} 话轮数:{len(dialog['turns'])}")

    # 保存结果（此处需补充人工评分后的最终筛选）
    # import json
    # with open(Config.output_path, 'w') as f:
    #     json.dump(final_selected, f, indent=2)