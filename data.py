"""
此代码用于将数据集处理为deepSeek使用的对话对，包含生成训练数据集和测试数据集。
格式为：
[
    {
        "messages": [
          {
            "role": "Tutor",
            "content": "Hi, could ......"
          },
          {
            "role": "Student",
            "content": "To tell......"
          }
        ]
    },....
]
"""

import json
import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """清洗文本，删除多余的空格和特殊字符，保持文本整洁。"""
    text = re.sub(r'\s+', ' ', text)  # 合并多余空格
    text = re.sub(r'(\u00A0|\u2028|\r\n)', ' ', text)  # 替换特殊空白符
    return text.strip()

def split_conversation(history: str) -> List[Dict]:
    """将原始对话字符串拆分为结构化轮次"""
    # 用于存储每一轮对话的信息，最终作为返回值。
    turns = []
    current_speaker = None
    current_text = []
    # 将整个对话按行分割，并逐行处理。
    for line in history.split('\n'):
        # 使用 clean_text 函数清洗每一行，去除多余的空格和特殊字符。如果清洗后的行为空，则跳过该行。
        line = clean_text(line)
        if not line:
            continue
        # 使用正则表达式检测行的开头是否包含说话者的名称（如 "Tutor" 或 "Student"）。
        # 如果没有检测到说话者，说明这是当前说话者的续发言，将其追加到 current_text 中。
        speaker_match = re.match(r'^(Tutor|Student|老师|学生)[：:]?\s*(.*)', line)
        if speaker_match:
            if current_speaker and current_text:  # 保存上一轮
                # 根据角色设置名称
                if current_speaker == "Tutor":
                    current_speaker = "assistant"
                elif current_speaker == "Student":
                    current_speaker = "user"

                turns.append({
                    "role": current_speaker,
                    "content": ' '.join(current_text)
                })
            current_speaker = speaker_match.group(1)
            current_text = [speaker_match.group(2)]
        else:
            current_text.append(line)

    # 在循环结束后，检查是否还有未保存的最后一轮对话，如果有，将其添加到 turns 列表中。
    if current_speaker and current_text:
        # 根据角色设置名称
        if current_speaker == "Tutor":
            current_speaker = "assistant"
        elif current_speaker == "Student":
            current_speaker = "user"
        turns.append({
            "role": current_speaker,
            "content": ' '.join(current_text)
        })
    return turns

def evaluate_response(item):
    """根据评分标准判断该回复的水平"""
    score = 0
    ML = item["Mistake_Location"]
    MI = item["Mistake_Identification"]
    PG = item["Providing_Guidance"]
    AC = item["Actionability"]
    # 为每个参数赋分
    if AC == "Yes":
        score += 2
    elif AC == "To some extent":
        score += 1
    elif AC == "No":
        score += 0

    if MI == "Yes":
        score += 2
    elif MI == "To some extent":
        score += 1
    elif MI == "No":
        score += 0

    if ML == "Yes":
        score += 2
    elif ML == "To some extent":
        score += 1
    elif ML == "No":
        score += 0

    if PG == "Yes":
        score += 2
    elif PG == "To some extent":
        score += 1
    elif PG == "No":
        score += 0

    return score

def train_data(input_file, output_file):
    """将数据集处理为训练数据格式"""
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    all_pairs = []  # 用于存储所有生成的对话对
    for item in raw_data:
        # 选出不同模型中最好的回复并添加到对话历史中
        best_responce = ""
        score = -1
        for text in item["tutor_responses"]:
            responce_score = evaluate_response(item["tutor_responses"][text]["annotation"])
            if responce_score > score:
                score = responce_score
                best_responce = item["tutor_responses"][text]["response"]

        responce = item["conversation_history"] + " \n Tutor:" + best_responce

        # 将对话历史拆分为结构化的轮次
        text = {
            "messages": split_conversation(responce)
        }
        all_pairs.append(text)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"生成完成！共处理 {len(raw_data)} 个对话，得到 {len(all_pairs)} 个训练对。")

def test_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    all_pairs = []  # 用于存储所有生成的对话对
    for item in raw_data:
        responce = item["conversation_history"]
        # 将对话历史拆分为结构化的轮次
        text = {
            "messages": split_conversation(responce)
        }
        all_pairs.append(text)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"生成完成！共处理 {len(raw_data)} 个对话，得到 {len(all_pairs)} 个训练对。")

# 主处理流程
def main():
    # 训练配置
    train_file = "data/mrbench_v3_devset.json"
    train_output = "data/train_deepseek.json"
    # 测试配置
    test_file = "data/mrbench_v3_testset.json"
    test_output = "data/test_deepseek.json"

    train_data(train_file, train_output)
    test_data(test_file, test_output)

if __name__ == "__main__":
    main()